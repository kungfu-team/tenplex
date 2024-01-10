package main

import (
	"flag"
	"fmt"
	golog "log"
	"os"
	"path"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/listflag"
	"github.com/kungfu-team/tenplex/tenplex-run/runop"
)

type MDPConfig = job.ParallelismConfig

type TrainConfig struct {
	ModelName      string
	ModelSize      string
	Dataset        string
	BatchSize      int
	MicroBatchSize int
}

type Run struct {
	// MdpConf   MDPConfig
	TrainConf TrainConfig
	Schedule  job.Schedule
	Central   bool
	Redeploy  bool
}

// func (r Run) MdpConf() MDPConfig {
// 	return *r.Schedule[0].ParaConf
// }

func genJobConf(r *Run) *job.JobConfig {
	return &job.JobConfig{
		Framework:      "megatron-lm",
		Precision:      "fp16",
		BatchSize:      r.TrainConf.BatchSize,
		MicroBatchSize: r.TrainConf.MicroBatchSize,
		SequenceLength: 1024,
		Dataset: ds.Dataset{
			Name:     "enwiki",
			IndexURL: "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt",
		},
		Image:         *image,
		Model:         r.TrainConf.ModelName,
		ModelSize:     r.TrainConf.ModelSize,
		TenplexPrefix: path.Join(`/home`, *user, `.tenplex`),
		Cluster: cluster.Cluster{
			GPUsPerHost:      4,
			GPUsPerContainer: 4,
			Hosts:            *hosts,
		},
		// SchedulerIP: "10.10.10.10",
		Schedule: r.Schedule,
		MLFSPort: 20010,
		User:     *user,
		Central:  r.Central,
		Redeploy: r.Redeploy,
	}
}

func genRuns(trains []TrainConfig, scheduleFiles []string, isCentral []bool) []Run {
	var runs []Run
	for _, sch := range scheduleFiles {
		sch, err := job.LoadFile(sch)
		if err != nil {
			panic(err)
		}
		for _, t := range trains {
			for _, central := range isCentral {
				r := Run{
					TrainConf: t,
					Schedule:  sch,
					Central:   central,
					Redeploy:  *redeploy,
				}
				runs = append(runs, r)
			}
		}
	}
	return runs
}

func genMDPs(sizes []int) []MDPConfig {
	var mdps []MDPConfig
	for _, s := range sizes {
		for pp := 1; pp <= s; pp++ {
			for mp := 1; mp <= s; mp++ {
				dp := s / (pp * mp)
				if pp*mp*dp == s {
					mdps = append(mdps,
						MDPConfig{
							PPSize: pp,
							MPSize: mp,
							Size:   s,
						})
				}
			}
		}
	}
	return mdps
}

func genTrainings(modelSizes []string, batchSizes []int, microBatchSizes []int) []TrainConfig {
	var trains []TrainConfig
	for _, m := range modelSizes {
		for _, b := range batchSizes {
			for _, mb := range microBatchSizes {
				trains = append(trains,
					TrainConfig{"gpt", m, "enwiki", b, mb})
			}
		}
	}
	return trains
}

var (
	user  = flag.String(`u`, os.Getenv(`USER`), ``)
	image = flag.String(`image`, ``, ``)

	hosts           = listflag.String("hosts", nil, "comma separated list of hosts")
	scheduleFiles   = listflag.String("schedule", nil, "comma separated list of file names")
	modelSizes      = listflag.String("model-sizes", nil, "comma separated list of file model sizes: medium | large | xl | 2.7B | 6.7B")
	batchSizes      = listflag.Int("batch-sizes", nil, `comma separated list of ints`)
	microBatchSizes = listflag.Int("micro-batch-sizes", nil, `comma separated list of ints`)
	dryrun          = flag.Bool(`dryrun`, false, ``)
	redeploy        = flag.Bool(`redeploy`, false, ``)
)

var log = golog.New(os.Stderr, `[multi-run] `, 0)

func main() {
	flag.Parse()

	log.Printf("Using %d hosts: %q", len(*hosts), *hosts)
	log.Printf("Using %d schedules: %q", len(*scheduleFiles), *scheduleFiles)
	log.Printf("Using %d model sizes: %q", len(*modelSizes), *modelSizes)
	log.Printf("Using %d batch sizes: %v", len(*batchSizes), *batchSizes)
	log.Printf("Using %d micro batch sizes: %v", len(*microBatchSizes), *microBatchSizes)

	var (
		isCentral = []bool{
			false,
			true,
		}
		trains = genTrainings(*modelSizes, *batchSizes, *microBatchSizes)
		runs   = genRuns(trains, *scheduleFiles, isCentral)
	)

	log.Printf("will run %d experiments", len(runs))

	runAll(runs)
}

func runAll(runs []Run) {
	t0 := time.Now()
	defer func() { log.Printf("Multi experiment took %s", time.Since(t0)) }()

	for i, r := range runs {
		n := logName("logs", fmt.Sprintf("%04d", i+1), r.TrainConf.ModelName, r.TrainConf.ModelSize, r.TrainConf.Dataset)
		func() {
			defer func() {
				if err := recover(); err != nil {
					log.Panicf("recovered %s", n)
				}
			}()
			runOne(n, r)
		}()
		log.Printf("finished %d/%d, took %s", i+1, len(runs), time.Since(t0))
	}
}

func runOne(n string, r Run) {
	jc := genJobConf(&r)
	if *dryrun {
		log.Printf("would run %s", n)
		return
	}
	runop.Main(jc)
	err := os.Rename("logs", n)
	if err != nil {
		log.Panic(err)
	}
}

func logName(ss ...string) string { return strings.Join(ss, `-`) + `.log` }
