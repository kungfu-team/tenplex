package runop

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/tenplex-run/counter"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/lgarithm/proc"
)

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

func RunTraining(jobConf *job.JobConfig, paraConf *job.ParallelismConfig, progress, maxStep int, jobID string, hosts []string) error {
	if !jobConf.NoTenplex {
		// add dataset to MLFS
		dpSize := paraConf.Size / (paraConf.PPSize * paraConf.MPSize)
		addDataStart := time.Now()
		if err := addDataset(dpSize, progress, jobConf, jobID); err != nil {
			log.Printf("add dataset failed but IGNORE: %v", err)
			// return err
		}
		log.Printf("Adding dataset took %s", time.Since(addDataStart))
	}

	// train
	log.Printf("Start training: %s", jobID)
	cluster := createCluster(jobConf, paraConf, hosts, maxStep, jobID)
	err := RunTrainMLMGo(cluster, jobConf)
	log.Printf("Finished training: %s", jobID)
	if err != nil {
		return err
	}
	return nil
}

func repartition(
	from, to *job.ParallelismConfig,
	step int,
	jobConf *job.JobConfig,
	jobID string,
) error {
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
			"--jobid", jobID,
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

func genJobID() string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("hello world %d", time.Now().Unix())))
	byt := h.Sum(nil)
	he := hex.EncodeToString(byt)
	return he[0:10]
}

func ScalingTraining(jobConf *job.JobConfig) {
	schedule := jobConf.Schedule
	jobID := genJobID()

	for i, scalingPoint := range schedule {
		// stop training
		if scalingPoint.ParaConf.Size == 0 {
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

		if scalingPoint.Step > 0 {
			if jobConf.Failure > 0 {
				log.Printf("Simulating failure")
				if err := simulateFailures(jobConf, jobID, jobConf.Failure); err != nil {
					log.Printf("simulateFailures: %v", err)
					return
				}
			}

			if !jobConf.NoTenplex {
				// repartition
				log.Printf("Start repartition func")
				err := repartition(
					schedule[i-1].ParaConf,
					scalingPoint.ParaConf,
					scalingPoint.Step,
					jobConf,
					jobID,
				)
				if err != nil {
					log.Printf("%v", err)
					return
				}
				log.Printf("Finished repartition func")
			}
		}

		maxStep := schedule[i+1].Step
		progress := scalingPoint.Step * (jobConf.BatchSize)
		hosts := jobConf.Cluster.Hosts
		if jobConf.Redeploy {
			if scalingPoint.Step == 0 {
				hosts = hosts[:len(hosts)/2]
			} else {
				hosts = hosts[len(hosts)/2:]
			}
		}
		if err := RunTraining(jobConf, scalingPoint.ParaConf, progress, maxStep, jobID, hosts); err != nil {
			// log.Panic(err)
			log.Printf("Training failed. Stopping containers")
			StopContainers(jobConf.Cluster.Hosts, jobConf.User)
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
