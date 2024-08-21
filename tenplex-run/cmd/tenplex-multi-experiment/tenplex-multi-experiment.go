package main

import (
	"flag"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
        "crypto/sha256"
	"encoding/hex"

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
	ID        string
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
		Central:     r.Central,
		Redeploy:    r.Redeploy,
		ParaConfigs: cfg.ParaConfigs,
		Seed:        1234,
	}
}

var str = strconv.Itoa

func genRuns(trains []TrainConfig, scheduleFiles []string, isCentral []bool) []Run {
	var runs []Run
	for _, sch := range scheduleFiles {
		sch, err := para_config.LoadScheduleFile(sch)
		if err != nil {
			panic(err)
		}
		for _, t := range trains {
			for _, central := range isCentral {
                                id := t.ModelName + `-` + t.ModelSize
                                if cfg.Redeploy {
                                        id = id + `-redeploy`
                                }
                                if central {
                                        id = id + `-central`
                                }
				r := Run{
					TrainConf: t,
					Schedule:  sch,
					Central:   central,
					Redeploy:  cfg.Redeploy,
					ID:        id,
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

	Hosts           listflag.Strings `flag:"hosts"`
	ScheduleFiles   listflag.Strings `flag:"schedule"`
	ModelSizes      listflag.Strings `flag:"model-sizes"`
	BatchSizes      listflag.Ints    `flag:"batch-sizes"`
	MicroBatchSizes listflag.Ints    `flag:"micro-batch-sizes"`

	DryRun   bool `flag:"dryrun"`
	Redeploy bool `flag:"redeploy"`
	Central  bool `flag:"central"`
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

	log.Printf("Using %d hosts: %q", len(cfg.Hosts), cfg.Hosts)
	log.Printf("Using %d schedules: %q", len(cfg.ScheduleFiles), cfg.ScheduleFiles)
	log.Printf("Using %d model sizes: %q", len(cfg.ModelSizes), cfg.ModelSizes)
	log.Printf("Using %d batch sizes: %v", len(cfg.BatchSizes), cfg.BatchSizes)
	log.Printf("Using %d micro batch sizes: %v", len(cfg.MicroBatchSizes), cfg.MicroBatchSizes)

	isCentral := []bool{false}
	if cfg.Central {
		isCentral = append(isCentral, true)
	}
	trains := genTrainings(cfg.ModelSizes, cfg.BatchSizes, cfg.MicroBatchSizes)
	runs := genRuns(trains, cfg.ScheduleFiles, isCentral)

	log.Printf("will run %d experiments", len(runs))

	runAll(runs)
}

func genRandomStr() string {
	currentTime := time.Now().Unix()
	hash := sha256.New()
	hash.Write([]byte(str(int(currentTime))))
	hashInBytes := hash.Sum(nil)
	hashString := hex.EncodeToString(hashInBytes)
        return hashString[:8]
}

func runAll(runs []Run) {
	t0 := time.Now()
	defer func() { log.Printf("Multi experiment took %s", time.Since(t0)) }()

	for i, r := range runs {
		logParts := []string{"logs", genRandomStr(), r.TrainConf.ModelName, r.TrainConf.ModelSize, cfg.Dataset.Name}
		if r.Central {
			logParts = append(logParts, "central")
		}
		n := strings.Join(logParts, `-`)
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
	runop.Main(jc, runop.Options{})
	err := os.Rename("logs", n)
	if err != nil {
		log.Panic(err)
	}
}
