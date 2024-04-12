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
	"github.com/lgarithm/go/tr"
)

type TrainConfig struct {
	ModelName      string
	ModelSize      string
	BatchSize      int
	MicroBatchSize int
}

func (tc TrainConfig) String() string {
	return fmt.Sprintf("%s[%s]/%d/%d", tc.ModelName, tc.ModelSize, tc.BatchSize, tc.MicroBatchSize)
}

func (tc TrainConfig) ID() string { // a valid filename
	return fmt.Sprintf("%s-%s-%d-%d", tc.ModelName, tc.ModelSize, tc.BatchSize, tc.MicroBatchSize)
}

type Run struct {
	TrainConf   TrainConfig
	Schedule    para_config.Schedule
	ParaConfigs para_config.ParaConfig
	ID          string
}

func (r Run) String() string {
	return fmt.Sprintf("#%s %s %s %s", r.ID, r.TrainConf, r.Schedule, r.ParaConfigs)
}

func toJobConf(r *Run) *job.JobConfig {
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
		Schedule:    r.Schedule,
		MLFSPort:    cfg.MLFSPort,
		User:        cfg.User,
		ParaConfigs: r.ParaConfigs,
		Seed:        1234,
	}
}

func ptr[T any](x T) *T { return &x }

func oneStageSchedule(size int) para_config.Schedule {
	return para_config.Schedule{
		{
			Step: ptr(0),
			Size: size,
		},
		{
			Step: ptr(50),
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
				ID: join(t.ID(), pc.ID()),
			}
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
				tc := TrainConfig{
					ModelName:      cfg.Model,
					ModelSize:      modelSize,
					BatchSize:      batchSize,
					MicroBatchSize: uBatchSize,
				}
				trains = append(trains, tc)
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

func (c MultiRunConfig) Print() {
	log.Printf("Using %d hosts: %q", len(c.Hosts), c.Hosts)
	log.Printf("Using %d model sizes: %q", len(c.ModelSizes), c.ModelSizes)
	log.Printf("Using %d batch sizes: %v", len(c.BatchSizes), c.BatchSizes)
	log.Printf("Using %d micro batch sizes: %v", len(c.MicroBatchSizes), c.MicroBatchSizes)
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
	cfg.Print()
	var (
		trains = genTrainings(cfg.ModelSizes, cfg.BatchSizes, cfg.MicroBatchSizes)
		runs   = genRuns(trains, cfg.MDPSizes)
	)
	runAll(runs)
}

func runAll(runs []Run) {
	defer tr.Patient(`run-all`, 300*time.Second).Done()
	log.Printf("will run %d experiments", len(runs))
	t0 := time.Now()
	defer func() { log.Printf("Multi experiment took %s", time.Since(t0)) }()

	var failed int
	for i, r := range runs {
		t1 := time.Now()
		n := logName("logs", fmt.Sprintf("%04d", i+1), r.ID)
		func() {
			defer func() {
				if err := recover(); err != nil {
					failed++
					log.Panicf("recovered %s", n)
				}
			}()
			runOne(n, r)
		}()
		log.Printf("finished %d/%d, %d failed, took %s (%s acc)", i+1, len(runs), failed, time.Since(t1), time.Since(t0))
	}
	log.Printf("finished %d, %d failed, total took: %s", len(runs), failed, time.Since(t0))
}

func runOne(n string, r Run) {
	defer tr.Patient(`runOne(`+n+`)`, 10*time.Second).Done()
	log.Printf("%s(%s, %s)...", `runOne`, n, r)
	jc := toJobConf(&r)
	if cfg.DryRun {
		log.Printf("would run %s", n)
		return
	}
	runop.Main(jc)
	os.RemoveAll(n)
	if err := os.Rename("logs", n); err != nil {
		log.Panic(err)
	}
	log.Printf("%s(%s, %s).", `runOne`, n, r)
}

func logName(ss ...string) string { return join(ss...) + `.log` }

func join(ss ...string) string { return strings.Join(ss, `-`) }

func genMDPs(sizes []int) []para_config.ParallelismConfig {
	var mdps []para_config.ParallelismConfig
	for _, s := range sizes {
		for pp := 1; pp <= s; pp++ {
			for mp := 1; mp <= s; mp++ {
				// if pp == 1 || mp == 1 {
				// 	continue
				// }
				if dp := s / (pp * mp); pp*mp*dp == s {
					mdp := para_config.ParallelismConfig{
						PPSize: pp,
						MPSize: mp,
						Size:   s,
					}
					mdps = append(mdps, mdp)
				}
			}
		}
	}
	return mdps
}
