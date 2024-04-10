package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
	"github.com/kungfu-team/tenplex/tenplex-run/job"
	"github.com/kungfu-team/tenplex/tenplex-run/listflag"
	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
	"github.com/kungfu-team/tenplex/tenplex-run/runop"
	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

type TrainConfig struct {
	ModelName      string
	ModelSize      string
	BatchSize      int
	MicroBatchSize int
}

type Run struct {
	TrainConf   TrainConfig
	Schedule    para_config.Schedule
	ParaConfigs para_config.ParaConfig
	ID          string
}

func genJobConf(r *Run) *job.JobConfig {
	return &job.JobConfig{
		ID:             r.ID,
		Framework:      "megatron-lm",
		Precision:      "fp16",
		BatchSize:      r.TrainConf.BatchSize,
		MicroBatchSize: r.TrainConf.MicroBatchSize,
		SequenceLength: 1024,
		Dataset:        cfg.Dataset,
		Image:          cfg.Image,
		Model:          r.TrainConf.ModelName,
		ModelSize:      r.TrainConf.ModelSize,
		TenplexPrefix:  cfg.TenplexPrefix,
		Cluster: cluster.Cluster{
			GPUsPerHost:      4,
			GPUsPerContainer: 4,
			Hosts:            cfg.Hosts,
		},
		// SchedulerIP: "10.10.10.10",
		Schedule:    r.Schedule,
		MLFSPort:    cfg.MLFSPort,
		User:        cfg.User,
		ParaConfigs: r.ParaConfigs,
		Seed:        1234,
	}
}

var runID int
var str = strconv.Itoa

func ptr[T any](x T) *T { return &x }

func oneStageSchedule(size int) para_config.Schedule {
	return para_config.Schedule{
		{
			Step: ptr(50),
			Size: size,
		},
		{
			Step: ptr(100),
			Size: 0,
		},
	}
}

func genRuns(trains []TrainConfig, MDPSizes []int) []Run {
	var runs []Run
	pcs := genMDPs(MDPSizes)
	for _, t := range trains {
		for _, pc := range pcs {
			r := Run{
				TrainConf: t,
				Schedule:  oneStageSchedule(pc.Size),
				ParaConfigs: para_config.ParaConfig{
					pc.Size: pc,
				},
				ID: str(runID),
			}
			runID++
			runs = append(runs, r)
		}
	}
	return runs
}

func genTrainings(modelSizes []string, batchSizes []int, microBatchSizes []int) []TrainConfig {
	var trains []TrainConfig
	for _, modelSize := range modelSizes {
		for _, batchSize := range batchSizes {
			for _, uBatchSize := range microBatchSizes {
				trains = append(trains,
					TrainConfig{
						ModelName:      cfg.Model,
						ModelSize:      modelSize,
						BatchSize:      batchSize,
						MicroBatchSize: uBatchSize,
					})
			}
		}
	}
	return trains
}

type MultiRunConfig struct {
	User          string `flag:"user"`
	MLFSPort      int    `flag:"mlfs-port"`
	TenplexPrefix string `flag:"tenplex-prefix"`
	Model         string `flag:"model"`
	Image         string `flag:"image"`
	Dataset       ds.Dataset

	Hosts           listflag.Strings `flag:"hosts"`
	ModelSizes      listflag.Strings `flag:"model-sizes"`
	BatchSizes      listflag.Ints    `flag:"batch-sizes"`
	MicroBatchSizes listflag.Ints    `flag:"micro-batch-sizes"`
	MDPSizes        listflag.Ints    `flag:"mdp-sizes"`

	DryRun bool `flag:"dryrun"`
}

var cfg MultiRunConfig

func init() {
	log.SetPrefix(`[fig-3] `)
	log.SetFlags(0)
}

func main() {
	structflag.RegisterFlags(&cfg, flag.CommandLine)
	structflag.RegisterFlags(&cfg.Dataset, flag.CommandLine)
	flag.Parse()
	// cfg.ParseParaConfig()

	log.Printf("Using %d hosts: %q", len(cfg.Hosts), cfg.Hosts)
	log.Printf("Using %d model sizes: %q", len(cfg.ModelSizes), cfg.ModelSizes)
	log.Printf("Using %d batch sizes: %v", len(cfg.BatchSizes), cfg.BatchSizes)
	log.Printf("Using %d micro batch sizes: %v", len(cfg.MicroBatchSizes), cfg.MicroBatchSizes)

	var (
		trains = genTrainings(cfg.ModelSizes, cfg.BatchSizes, cfg.MicroBatchSizes)
		runs   = genRuns(trains, cfg.MDPSizes)
	)

	log.Printf("will run %d experiments", len(runs))

	runAll(runs)
}

func runAll(runs []Run) {
	t0 := time.Now()
	defer func() { log.Printf("Multi experiment took %s", time.Since(t0)) }()

	for i, r := range runs {
		n := logName("logs", fmt.Sprintf("%04d", i+1), r.TrainConf.ModelName, r.TrainConf.ModelSize, cfg.Dataset.Name)
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
	log.Printf("%s(%s, ?)", `runOne`, n)
	jc := genJobConf(&r)
	if cfg.DryRun {
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

func genMDPs(sizes []int) []para_config.ParallelismConfig {
	var mdps []para_config.ParallelismConfig
	for _, s := range sizes {
		for pp := 1; pp <= s; pp++ {
			for mp := 1; mp <= s; mp++ {
				dp := s / (pp * mp)
				if pp*mp*dp == s {
					mdps = append(mdps,
						para_config.ParallelismConfig{
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
