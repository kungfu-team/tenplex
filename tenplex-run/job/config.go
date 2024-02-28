package job

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"path"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/tenplex-run/cluster"
	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
)

type JobConfig struct {
	ID                string
	Framework         string
	Precision         string
	BatchSize         int
	MicroBatchSize    int
	SequenceLength    int
	Dataset           ds.Dataset
	Image             string
	Model             string
	ModelSize         string
	TenplexPrefix     string
	Cluster           cluster.Cluster
	SchedulerEndpoint string
	scheduleFile      string
	Schedule          para_config.Schedule
	MLFSPort          int
	User              string
	DockerNetwork     string
	Failure           int
	Central           bool
	Redeploy          bool
	NoTenplex         bool
	TimeBased         bool
	ParaConfigFile    string
	ParaConfigs       para_config.ParaConfig
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
	flag.StringVar(&j.scheduleFile, "schedule-file", "", "Schedule file path")
	flag.IntVar(&j.MLFSPort, "mlfs-port", 0, "MLFS port")
	flag.StringVar(&j.User, "user", "kungfu", "Remote host user")
	flag.StringVar(&j.DockerNetwork, "network", "tenplex", "Docker network name")
	flag.IntVar(&j.Failure, "failure", 0, "Number of host failures to simulate")
	flag.BoolVar(&j.Central, "central", false, "Set to true to transfrom state centrally")
	flag.BoolVar(&j.Redeploy, "redeploy", false, "Set to true to redeploy job")
	flag.BoolVar(&j.NoTenplex, "no-tenplex", false, "Set to true to run without Tenplex")
	flag.BoolVar(&j.TimeBased, "time-based", false, "Set to true to run scaling based on time")
	flag.StringVar(&j.ParaConfigFile, "para-config", "", "Set Para config file")
	j.Cluster.RegisterFlags(flag)
}

func (j *JobConfig) ParseParaConfig() {
	if len(j.ParaConfigFile) > 0 {
		paraConfigs, err := para_config.LoadFile(j.ParaConfigFile)
		if err != nil {
			log.Panic(err)
		}
		j.ParaConfigs = paraConfigs
		log.Printf("ParaConfigs start")
		for size, para := range j.ParaConfigs {
			log.Printf("%d: %v", size, para)
		}
		log.Printf("ParaConfigs end")
		return
	}
	j.ParaConfigs = para_config.GenParaConfig()
}

func (j *JobConfig) ParseSchedule() {
	j.Schedule = para_config.GenSchedule(j.scheduleFile)
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

func (j *JobConfig) DistFlags(c MDPConfig, rank int) []string {
	return []string{
		`--nproc_per_node`, str(c.GPUPerNode),
		`--nnodes`, str(c.NumNodes),
		`--node_rank`, str(rank),
		`--master_addr`, j.Cluster.Hosts[0],
		`--master_port`, `6000`,
	}
}

func (j *JobConfig) LogFlags(c MDPConfig) []string {
	return []string{
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters`, `0`, // default: 10
	}
}

func (j *JobConfig) TenplexFlags(c MDPConfig, host string) []string {
	if j.NoTenplex {
		return nil
	}
	var cmd []string
	cmd = append(cmd, `--tenplex`)
	cmd = append(cmd, `--mlfs-path`, `/data/mlfs`)
	cmd = append(cmd, `--jobid`, j.ID)
	cmd = append(cmd, `--host-ip`, host)
	cmd = append(cmd, `--mlfs-port`, str(j.MLFSPort))
	return cmd
}

func (j *JobConfig) OtherFlags(c MDPConfig) []string {
	const checkpoint_path = `/data/ckpt`
	var cmd []string
	args := []string{
		`--save`, checkpoint_path,
		`--load`, checkpoint_path,
		`--tensor-model-parallel-size`, str(c.ModelParallelSize),
		`--pipeline-model-parallel-size`, str(c.PipelineParallelSize),
		`--tensorboard-dir`, path.Join(checkpoint_path, `tensorboard`),
	}
	cmd = append(cmd, args...)
	if len(j.SchedulerEndpoint) > 0 {
		cmd = append(cmd, `--scheduler-addr`, j.SchedulerEndpoint)
	}
	if c.Precision == "fp16" {
		cmd = append(cmd, `--fp16`)
	}
	return cmd
}
