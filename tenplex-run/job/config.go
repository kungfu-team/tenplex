package job

import (
	"flag"

	"github.com/kungfu-team/mlfs/ds"
	"github.com/kungfu-team/tenplex-run/cluster"
)

type JobConfig struct {
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
	Schedule       Schedule
	MLFSPort       int
	User           string
	DockerNetwork  string
	Failure        int
	Central        bool
	Redeploy       bool
}

type ParallelismConfig struct {
	Size   int `json:"size"`
	PPSize int `json:"pp_size"`
	MPSize int `json:"mp_size"`
}

var scheduleFile string

func (j *JobConfig) RegisterFlags(flag *flag.FlagSet) {
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
	flag.StringVar(&scheduleFile, "schedule-file", "", "Schedule file path")
	flag.IntVar(&j.MLFSPort, "mlfs-port", 0, "MLFS port")
	flag.StringVar(&j.User, "user", "kungfu", "Remote host user")
	flag.StringVar(&j.DockerNetwork, "network", "tenplex", "Docker network name")
	flag.IntVar(&j.Failure, "failure", 0, "Number of host failures to simulate")
	flag.BoolVar(&j.Central, "central", false, "Set to true to transfrom state centrally")
	flag.BoolVar(&j.Redeploy, "redeploy", false, "Set to true to redeploy job")
	j.Cluster.RegisterFlags(flag)
}

func (j *JobConfig) ParseSchedule() {
	j.Schedule = GenSchedule(scheduleFile)
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
