package job

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/mlfs"
	"github.com/kungfu-team/tenplex/state_transformer/meta"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/runop"
	"github.com/lgarithm/proc"
)

var (
	ssh   = proc.SSH
	par   = proc.Par
	run   = proc.Run
	term  = proc.Term
	stdio = proc.Stdio
)

type (
	P    = job.P
	Proc = job.Proc
)

type Runner struct {
	Job           Job
	JobConfig     *job.JobConfig
	Finished      bool
	Cluster       *cluster.Cluster
	ParaConfig    *job.ParallelismConfig
	CurStep       int
	MLFSPort      int
	TenplexPrefix string
}

func (ru *Runner) queryIter() error {
	pa := fmt.Sprintf("job/%s/iter", ru.Job.ID)
	url := fmt.Sprintf("http://%s:%d/query", ru.Cluster.Hosts[0], ru.MLFSPort)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		log.Printf("new request error")
		return err
	}
	query := req.URL.Query()
	query.Add("path", pa)
	req.URL.RawQuery = query.Encode()
	client := http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("client do error")
		return err
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		log.Printf("readall error")
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("response failed with status code: %d", resp.StatusCode)
	}
	iter, err := strconv.Atoi(string(body))
	if err != nil {
		log.Printf("conv error")
		return err
	}

	log.Printf("queryIter %s iter %d", ru.Job.ID, iter)
	ru.CurStep = iter
	log.Printf("queryIter  %s ru.CurStep %d", ru.Job.ID, ru.CurStep)

	return nil
}

func (ru *Runner) RunTraining(wg *sync.WaitGroup, ch *chan string, schedulerAddr string) {
	ru.JobConfig = &job.JobConfig{
		Framework:      ru.Job.Framework,
		Precision:      ru.Job.Precision,
		BatchSize:      ru.Job.BatchSize,
		MicroBatchSize: ru.Job.MicroBatchSize,
		SequenceLength: ru.Job.SequenceLength,
		Dataset:        ru.Job.Dataset,
		Image:          ru.Job.Image,
		Model:          ru.Job.Model,
		ModelSize:      ru.Job.ModelSize,
		TenplexPrefix:  ru.TenplexPrefix,
		Cluster:        *ru.Cluster,
		SchedulerIP:    schedulerAddr,
		MLFSPort:       mlfs.DefaultCtrlPort,
		User:           "marcel", // TODO: make dynamic
		Failure:        ru.Job.Failure,
	}

	runop.PullImages(ru.JobConfig)

	progress := ru.CurStep * ru.JobConfig.BatchSize
	log.Printf("job config %+v", ru.JobConfig)
	log.Printf("para config %+v", ru.ParaConfig)
	log.Printf("progress %d", progress)
	log.Printf("steps %d", ru.Job.Steps)
	log.Printf("id %s", ru.Job.ID)
	func() {
		defer func() {
			if err := recover(); err != nil {
				log.Panicf("recovered from RunTraining: %v", err)
			}
		}()
		err := runop.RunTraining(ru.JobConfig, ru.ParaConfig, progress, ru.Job.Steps, ru.Job.ID, ru.Cluster.Hosts)
		if err != nil {
			log.Panicf("Run training failed. %v", err)
		}
	}()

	err := ru.queryIter()
	if err != nil {
		log.Panicf("cannot query iter %v", err)
	}

	if ru.CurStep >= ru.Job.Steps {
		ru.Finished = true
		*ch <- ru.Job.ID
	}

	wg.Done()
}

func genID() func() int {
	var id int
	return func() int { x := id; id++; return x }
}

var getTransformID = genID()

var str = strconv.Itoa
var home = os.Getenv(`HOME`)

func (ru *Runner) TransformStateWithCmd(conf *meta.Config, newNumDev int, newCluster *cluster.Cluster) error {
	transformPs := []P{}
	for i := 0; i < newNumDev; i++ {
		hostIdx := i / conf.GpusPerHost
		host := newCluster.Hosts[hostIdx] // new Hosts
		args := []string{
			"--ckpt-struct-dir", conf.CkptStructDir,
			"--precision", conf.Precision,
			"--input-timestamp", conf.InputTimestamp,
			"--output-timestamp", conf.OutputTimestamp,
			"--sequence-length", str(conf.SeqLength),
			"--source-pp-degree", str(conf.SourcePPDegree),
			"--target-pp-degree", str(conf.TargetPPDegree),
			"--source-mp-degree", str(conf.SourceMPDegree),
			"--target-mp-degree", str(conf.TargetMPDegree),
			"--source-size", str(conf.SourceSize),
			"--target-size", str(conf.TargetSize),
			"--target-rank", str(i),
			"--source-hosts", strings.Join(conf.SourceHosts, ","),
			"--target-hosts", strings.Join(conf.TargetHosts, ","),
			"--mdp-library", conf.MdpLibrary,
			"--gpus-per-host", str(conf.GpusPerHost),
			"--jobid", conf.JobID,
			"--num-layers", str(ru.Job.NumLayers),
			"--model", ru.Job.Model,
			"--model-size", ru.Job.ModelSize,
			"--vocab-size", str(ru.Job.VocabSize),
			"--step", str(ru.CurStep),
		}
		p := Proc{
			Prog: path.Join(home, ".tenplex/bin/tenplex-state-transformer"),
			Args: args,
			Host: host,
		}
		prefix := fmt.Sprintf("[%s %d Transform] ", host, i)
		transformPs = append(transformPs, term(prefix, ssh(p)))
	}
	r := run(job.Tee2Files(`logs/transform-`+str(getTransformID()), par(transformPs...)), &stdio)
	if r.Err != nil {
		return r.Err
	}

	return nil
}

func (ru *Runner) TransformState(newParaConf *job.ParallelismConfig, newSubCluster *cluster.Cluster) {
	log.Printf("TransformState ru.CurStep %d", ru.CurStep)
	conf := meta.Config{
		CkptStructDir:   path.Join(ru.TenplexPrefix, "transformer-checkpoint"),
		SourceMPDegree:  ru.ParaConfig.MPSize,
		TargetMPDegree:  newParaConf.MPSize,
		SourcePPDegree:  ru.ParaConfig.PPSize,
		TargetPPDegree:  newParaConf.PPSize,
		SourceSize:      ru.ParaConfig.Size,
		TargetSize:      newParaConf.Size,
		SourceDPDegree:  ru.ParaConfig.Size / (ru.ParaConfig.MPSize * ru.ParaConfig.PPSize),
		TargetDPDegree:  newParaConf.Size / (newParaConf.MPSize * newParaConf.PPSize),
		Precision:       ru.Job.Precision,
		OutputTimestamp: "load",
		InputTimestamp:  "save",
		SourceHosts:     ru.Cluster.Hosts,
		TargetHosts:     newSubCluster.Hosts,
		Port:            ru.MLFSPort,
		GpusPerHost:     newSubCluster.GPUsPerHost,
		MdpLibrary:      ru.Job.Framework,
		SeqLength:       ru.Job.SequenceLength,
		JobID:           ru.Job.ID,
		NumLayers:       ru.Job.NumLayers,
		Model:           ru.Job.Model,
		ModelSize:       ru.Job.ModelSize,
		VocabSize:       ru.Job.VocabSize,
		Step:            ru.CurStep,
	}

	newNumDev := newParaConf.Size
	if err := ru.TransformStateWithCmd(&conf, newNumDev, newSubCluster); err != nil {
		log.Panicf("TransformStateWithCmd failed: %v", err)
	}

	for _, host := range newSubCluster.Hosts {
		cl, err := mlfs.NewClientTo(host, ru.MLFSPort)
		if err != nil {
			log.Panicf("cannot create mlfs client to write iter")
		}
		cl.UploadReplace(fmt.Sprintf("job/%s/iter", ru.Job.ID), []byte(str(ru.CurStep)), true)
	}
}

func (ru *Runner) TransformAndRun(newParaConf *job.ParallelismConfig, newSubClu *cluster.Cluster, wg *sync.WaitGroup, ch *chan string, schedulerAddr string) {
	start := time.Now()
	log.Printf("TransformAndRun %s on new cluster: %s", ru.Job.ID, strings.Join(newSubClu.Hosts, ","))
	ru.TransformState(newParaConf, newSubClu)
	log.Printf("TransformAndRun %s on new cluster: %s | TransformState took %s", ru.Job.ID, strings.Join(newSubClu.Hosts, ","), time.Since(start))

	ru.Cluster = newSubClu
	ru.ParaConfig = newParaConf

	ru.RunTraining(wg, ch, schedulerAddr)
	log.Printf("TransformAndRun %s on new cluster: %s | done:RunTraining", ru.Job.ID, strings.Join(newSubClu.Hosts, ","))
}

var tee2Files = proc.Tee2Files
