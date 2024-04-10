package main

import (
	"flag"
	"fmt"
	"log"
	"os"
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
	TrainConf TrainConfig
	Schedule  para_config.Schedule
	Central   bool
	Redeploy  bool
}

func genJobConf(r *Run) *job.JobConfig {
	return &job.JobConfig{
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
			Hosts:            *hosts,
		},
		// SchedulerIP: "10.10.10.10",
		Schedule:    r.Schedule,
		MLFSPort:    cfg.MLFSPort,
		User:        cfg.User,
		Central:     r.Central,
		Redeploy:    r.Redeploy,
		ParaConfigs: cfg.ParaConfigs,
	}
}

func genRuns(trains []TrainConfig, scheduleFiles []string, isCentral []bool) []Run {
	var runs []Run
	for _, sch := range scheduleFiles {
		sch, err := para_config.LoadScheduleFile(sch)
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
	User           string `flag:"user"`
	MLFSPort       int    `flag:"mlfs-port"`
	TenplexPrefix  string `flag:"tenplex-prefix"`
	Model          string `flag:"model"`
	Image          string `flag:"image"`
	Dataset        ds.Dataset
	ParaConfigFile string `flag:"para-config"`
	ParaConfigs    para_config.ParaConfig
}

func (j *MultiRunConfig) ParseParaConfig() {
	var err error
	j.ParaConfigs, err = para_config.LoadFile(j.ParaConfigFile)
	if err != nil {
		log.Panicf("%s: %v", `ParseParaConfig`, err)
	}
	for i, size := range j.ParaConfigs.Sizes() {
		log.Printf("ParaConfig[%d/%d]: %s", i+1, len(j.ParaConfigs), j.ParaConfigs[size])
	}
}

var cfg MultiRunConfig

var (
	hosts           = listflag.String("hosts", nil, "comma separated list of hosts")
	scheduleFiles   = listflag.String("schedule", nil, "comma separated list of file names")
	modelSizes      = listflag.String("model-sizes", nil, "comma separated list of file model sizes: medium | large | xl | 2.7B | 6.7B")
	batchSizes      = listflag.Int("batch-sizes", nil, `comma separated list of ints`)
	microBatchSizes = listflag.Int("micro-batch-sizes", nil, `comma separated list of ints`)
	dryrun          = flag.Bool(`dryrun`, false, ``)
	redeploy        = flag.Bool(`redeploy`, false, ``)
)

// var log = golog.New(os.Stderr, `[multi-run] `, 0)

func init() {
	log.SetPrefix(`[multi-run] `)
	log.SetFlags(0)
}

func main() {
	structflag.RegisterFlags(&cfg, flag.CommandLine)
	structflag.RegisterFlags(&cfg.Dataset, flag.CommandLine)
	flag.Parse()
	cfg.ParseParaConfig()

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
