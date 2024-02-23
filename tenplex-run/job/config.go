package job

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
)

type JobConfig struct {
	ID             string
	Framework      string
	Precision      string
	BatchSize      int
	MicroBatchSize int
	SequenceLength int
	Dataset        ds.Dataset
	Image          string
	Model          string
	ModelSize      string
	TenplexPrefix  string
	Cluster        cluster.Cluster
	SchedulerIP    string
	scheduleFile   string
	Schedule       Schedule
	MLFSPort       int
	User           string
	DockerNetwork  string
	Failure        int
	Central        bool
	Redeploy       bool
	NoTenplex      bool
}

type ParallelismConfig struct {
	Size   int `json:"size"`
	PPSize int `json:"pp_size"`
	MPSize int `json:"mp_size"`
}

func genJobID() string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("hello world %d", time.Now().Unix())))
	byt := h.Sum(nil)
	he := hex.EncodeToString(byt)
	return he[0:10]
}

func (j *JobConfig) RegisterFlags(flag *flag.FlagSet) {
	flag.StringVar(&j.ID, "jobid", genJobID(), "job id")
	flag.StringVar(&j.Framework, "framework", "", "megatron-lm OR deepspeed")
	flag.StringVar(&j.Model, "model", "", "gpt OR bert")
	flag.StringVar(&j.ModelSize, "model-size", "", "model size")
	flag.StringVar(&j.Dataset.Name, "dataset", "", "enwiki OR openwebtext")
	flag.StringVar(&j.Dataset.IndexURL, "index-url", "", "dataset index URL")
	flag.StringVar(&j.Image, "image", "kungfu.azurecr.io/deepspeed-run:latest", "")
	flag.StringVar(&j.TenplexPrefix, "tenplex-prefix", "", "")
	flag.IntVar(&j.BatchSize, "batch-size", 0, "batch size")
	flag.IntVar(&j.MicroBatchSize, "micro-batch-size", 0, "micro batch size")
	flag.IntVar(&j.SequenceLength, "seq-length", 1024, "sequence length")
	flag.StringVar(&j.Precision, "precision", "", "fp32 OR fp16 OR bf16")
	flag.StringVar(&j.SchedulerIP, "scheduler-ip", "", "Scheduler IP")
	flag.StringVar(&j.scheduleFile, "schedule-file", "", "Schedule file path")
	flag.IntVar(&j.MLFSPort, "mlfs-port", 0, "MLFS port")
	flag.StringVar(&j.User, "user", "kungfu", "Remote host user")
	flag.StringVar(&j.DockerNetwork, "network", "tenplex", "Docker network name")
	flag.IntVar(&j.Failure, "failure", 0, "Number of host failures to simulate")
	flag.BoolVar(&j.Central, "central", false, "Set to true to transfrom state centrally")
	flag.BoolVar(&j.Redeploy, "redeploy", false, "Set to true to redeploy job")
	flag.BoolVar(&j.NoTenplex, "no-tenplex", false, "Set to true to run without Tenplex")
	j.Cluster.RegisterFlags(flag)
}

func (j *JobConfig) ParseSchedule() {
	j.Schedule = GenSchedule(j.scheduleFile)
}

func OverwriteHostIdx(hostIdx int, jc *JobConfig) int {
	if jc.Central {
		return 0
	}
	return hostIdx
}

func OverwriteHost(host string, jc *JobConfig) string {
	if jc.Central {
		return jc.Cluster.Hosts[0]
	}
	return host
}
