package runop

import (
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kungfu-team/tenplex/tenplex-run/counter"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
	// "github.com/kungfu-team/tenplex/tenplex-run/web"
	"github.com/lgarithm/proc"
)

const DefaultSchedulerPort = 22222

var (
	par   = proc.Par
	stdio = proc.Stdio
	run   = proc.Run
	pmap  = job.Pmap
	str   = strconv.Itoa
	ssh   = proc.SSH
	term  = proc.Term
)

type (
	P    = proc.P
	Proc = proc.Proc
)

func train(c *job.ContainerCluster, jobConf *job.JobConfig) error {
	if r := run(c.Stop(), &stdio); r.Err != nil {
		// log.Panic(r.Err)
		return r.Err
	}
	log.Printf("old containers cleaned")
	log.Printf("starting train workers")
	if r := run(c.RunTrain(jobConf), &stdio); r.Err != nil {
		// log.Panic(r.Err)
		return r.Err
	}
	return nil
}

func RunTraining(jobConf *job.JobConfig, paraConf *para_config.ParallelismConfig, progress, maxStep int, hosts []string) error {
	if !jobConf.NoTenplex {
		// add dataset to MLFS
		dpSize := paraConf.Size / (paraConf.PPSize * paraConf.MPSize)
		addDataStart := time.Now()
		if err := addDataset(dpSize, progress, jobConf); err != nil {
			log.Printf("add dataset failed but IGNORE: %v", err)
			// return err
		}
		log.Printf("Adding dataset with DP %d took %s", dpSize, time.Since(addDataStart))
	}

	// train
	log.Printf("Start training: %s", jobConf.ID)
	cluster := createCluster(jobConf, paraConf, hosts, maxStep)
	err := RunTrainMLMGo(cluster, jobConf)
	log.Printf("Finished training: %s", jobConf.ID)
	if err != nil {
		return err
	}
	return nil
}

func repartition(from, to *para_config.ParallelismConfig, step int, jobConf *job.JobConfig) error {
	var round = RoundID.Next()
	var home = path.Join("/home", jobConf.User)
	t0 := time.Now()
	defer func() { log.Printf("State transformation took %s", time.Since(t0)) }()
	// run state transformation
	var transformPs []P
	for i := 0; i < to.Size; i++ {
		hostIdx := i / jobConf.Cluster.GPUsPerHost
		hostIdx = job.OverwriteHostIdx(hostIdx, jobConf)
		host := jobConf.Cluster.Hosts[hostIdx]
		var args []string
		if jobConf.Framework == "megatron-lm" {
			args = []string{
				"--mdp-library", "megatron-lm",
			}
		} else if jobConf.Framework == "deepspeed" {
			args = []string{
				"--mdp-library", "deepspeed",
			}
		} else {
			return errors.New("framework not supported")
		}
		var numLayers int
		if jobConf.Model == "gpt" && (jobConf.ModelSize == "2.7B" || jobConf.ModelSize == "6.7B") {
			numLayers = 32
		} else if jobConf.Model == "bert" && jobConf.ModelSize == "base" {
			numLayers = 12
		} else {
			numLayers = 24
		}
		var srcHo string
		var trgHo string
		if jobConf.Redeploy {
			hosts := jobConf.Cluster.Hosts
			firstHalf := hosts[:len(hosts)/2]
			secondHalf := hosts[len(hosts)/2:]
			srcHo = strings.Join(firstHalf, ",")
			trgHo = strings.Join(secondHalf, ",")
		} else {
			srcHo = strings.Join(jobConf.Cluster.Hosts, ",")
			trgHo = srcHo
		}
		args = append(args,
			"--ckpt-struct-dir", path.Join(home, ".tenplex/transformer-checkpoint"),
			"--precision", "fp16",
			"--input-timestamp", "save",
			"--output-timestamp", "load",
			"--sequence-length", str(jobConf.SequenceLength),
			"--source-pp-degree", str(from.PPSize),
			"--target-pp-degree", str(to.PPSize),
			"--source-mp-degree", str(from.MPSize),
			"--target-mp-degree", str(to.MPSize),
			"--source-size", str(from.Size),
			"--target-size", str(to.Size),
			"--target-rank", str(i),
			"--source-hosts", srcHo,
			"--target-hosts", trgHo,
			"--jobid", jobConf.ID,
			"--gpus-per-host", "4",
			"--num-layers", str(numLayers),
			"--model", jobConf.Model,
			"--model-size", jobConf.ModelSize,
			"--vocab-size", str(30524), // TODO get from flag
			"--step", str(step),
		)
		if jobConf.Central {
			args = append(args, `--central`)
		}
		migrate := Proc{
			Prog: "tenplex-state-transformer",
			Args: args,
			Host: host,
			User: jobConf.User,
		}
		name := fmt.Sprintf("logs/tenplex-state-transformer-%d-%d", round, i)
		p := proc.Tee2Files(name, ssh(migrate))
		prefix := fmt.Sprintf("[%s %d] ", host, i)
		transformPs = append(transformPs, term(prefix, p))

		log.Printf("Rank %d state transformation with %v", i, args)
	}
	log.Printf("Run state transformation")
	r := run(par(transformPs...), &stdio)
	log.Printf("Finished state transformation")
	return r.Err
}

var RoundID = counter.New()

type StopServer struct {
	Stopped  int32
	FinishWG sync.WaitGroup
}

func (ss *StopServer) Start(port int) {
	mux := http.NewServeMux()
	mux.HandleFunc("/stop", ss.GetStop)
	hs := http.Server{
		Addr: fmt.Sprintf(":%d", port),
		// Handler: web.WithLogReq(mux),
		Handler: mux,
	}
	go hs.ListenAndServe()
}

func (ss *StopServer) GetStop(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		log.Printf("method must be get")
		http.Error(w, "method must be get", http.StatusMethodNotAllowed)
		return
	}
	if atomic.LoadInt32(&ss.Stopped) > 0 {
		w.Write([]byte("stop"))
	} else {
		w.Write([]byte("run"))
	}
}

func ScalingTraining(jobConf *job.JobConfig) {
	var stopSer StopServer
	schedule := jobConf.Schedule
	if jobConf.TimeBased {
		port := DefaultSchedulerPort
		stopSer.Start(port)
	}

	for i, scalingPoint := range schedule {
		newPara := jobConf.ParaConfigs[scalingPoint.Size]
		// stop training
		if scalingPoint.Size == 0 {
			log.Printf("Stopping training")
			return
		}

		// redundancy in failure scenario
		if jobConf.Failure > 0 {
			if err := setRedundancy(jobConf); err != nil {
				log.Printf("%v", err)
				return
			}
		}

		// if scalingPoint.Step > 0 {
		if i > 0 {
			if jobConf.Failure > 0 {
				log.Printf("Simulating failure")
				if err := simulateFailures(jobConf, jobConf.Failure); err != nil {
					log.Printf("simulateFailures: %v", err)
					return
				}
			}

			if !jobConf.NoTenplex {
				// repartition
				log.Printf("Start repartition func")
				curPara := jobConf.ParaConfigs[schedule[i-1].Size]
				err := repartition(
					&curPara,
					&newPara,
					getStep(jobConf, scalingPoint),
					jobConf,
				)
				if err != nil {
					log.Printf("%v", err)
					return
				}
				log.Printf("Finished repartition func")
			}
		}

		maxStep := getMaxStep(i, jobConf.TimeBased, schedule)
		hosts := jobConf.Cluster.Hosts
		if jobConf.Redeploy {
			if i == 0 {
				hosts = hosts[:len(hosts)/2]
			} else {
				hosts = hosts[len(hosts)/2:]
			}
		}
		progress := getStep(jobConf, scalingPoint) * jobConf.BatchSize
		if jobConf.TimeBased {
			stopSer.FinishWG.Add(1)
			go func() {
				if err := RunTraining(jobConf, &newPara, progress, maxStep, hosts); err != nil {
					log.Printf("Training failed. Stopping containers")
					StopContainers(jobConf.Cluster.Hosts, jobConf.User)
				}
				stopSer.FinishWG.Done()
			}()
			d := time.Duration(*scalingPoint.Time) * time.Minute
			log.Printf("sleep for %s", d)
			time.Sleep(d)
			log.Printf("woke up again")

			// stop training
			atomic.StoreInt32(&stopSer.Stopped, 1)
			stopSer.FinishWG.Wait()
			atomic.StoreInt32(&stopSer.Stopped, 0)
		} else {
			if err := RunTraining(jobConf, &newPara, progress, maxStep, hosts); err != nil {
				log.Printf("Training failed. Stopping containers")
				StopContainers(jobConf.Cluster.Hosts, jobConf.User)
			}
		}
	}
}

func runP(p P, finish chan string) {
	r := run(p, &stdio)
	if r.Err != nil {
		// log.Panicf("running P failed with %v", r.Err)
		log.Printf("running P failed with %v", r.Err)
	}
	finish <- "finished"
}

func RunTrainMLMGo(c *job.ContainerCluster, jobConf *job.JobConfig) error {
	stageID := job.GetStageId()
	workers := c.Workers

	mkMountDirStart := time.Now()
	r := run(par(job.Cmap(c.MkMountDirs, workers...)...), &stdio)
	if r.Err != nil {
		log.Printf("make mount directories failed")
		return r.Err
	}
	log.Printf("Making mount directories took %s", time.Since(mkMountDirStart))

	finish := make(chan string)
	for i, w := range workers {
		p := c.Run(w)
		p = job.Tee2Files(fmt.Sprintf("logs/stage-%02d-worker-%02d", stageID, i), p)
		go runP(p, finish)
	}

	<-finish
	StopContainers(jobConf.Cluster.Hosts, jobConf.User)

	return nil
}

func getStep(j *job.JobConfig, sp para_config.ScalingPoint) int {
	if j.TimeBased {
		step, err := QueryIter(j.ID, j.Cluster.Hosts[0], j.MLFSPort)
		if err != nil {
			log.Printf("QueryIter failed: %v", err)
			return 0
		}
		return step
	}
	return *sp.Step
}

func getMaxStep(i int, timeBased bool, schedule para_config.Schedule) int {
	if timeBased {
		return 10000
	} else {
		return *schedule[i+1].Step
	}
}

func QueryIter(jobID string, host string, port int) (int, error) {
	query := url.Values{}
	query.Set("path", fmt.Sprintf("job/%s/iter", jobID))
	u := url.URL{
		Scheme:   `http`,
		Host:     net.JoinHostPort(host, str(port)),
		Path:     `/query`,
		RawQuery: query.Encode(),
	}
	url := u.String()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		log.Printf("new request error")
		return 0, err
	}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("client do error")
		return 0, err
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		log.Printf("readall error")
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("response failed with status code: %d", resp.StatusCode)
	}
	iter, err := strconv.Atoi(string(body))
	if err != nil {
		log.Printf("conv error")
		return 0, err
	}
	return iter, nil
}
