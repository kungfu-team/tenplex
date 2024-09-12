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
	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

type JobConfig struct {
	BatchSize         int  `flag:"batch-size"`
	Central           bool `flag:"central"`
	Cluster           cluster.Cluster
	Dataset           ds.Dataset
	DockerNetwork     string `flag:"network"`
	Failure           int    `flag:"failure"`
	Framework         string `flag:"framework"`
	ID                string `flag:"jobid"`
	Image             string `flag:"image"`
	MicroBatchSize    int    `flag:"micro-batch-size"`
	MLFSPort          int    `flag:"mlfs-port"`
	Model             string `flag:"model"`
	ModelSize         string `flag:"model-size"`
	NoTenplex         bool   `flag:"no-tenplex"`
	ParaConfigFile    string `flag:"para-config"`
	ParaConfigs       para_config.ParaConfig
	Precision         string `flag:"precision"`
	Redeploy          bool   `flag:"redeploy"`
	Schedule          para_config.Schedule
	scheduleFile      string `flag:"schedule-file"`
	SchedulerEndpoint string
	SequenceLength    int    `flag:"seq-length" default:"1024"`
	TenplexPrefix     string `flag:"tenplex-prefix"`
	TimeBased         bool   `flag:"time-based"`
	User              string `flag:"user"`
	Seed              int    `flag:"seed" default:"1234"`
	NoShuffle         bool   `flag:"no-shuffle" default:"false"`
	GenParaConfig     bool   `flag:"gen-para-config" default:"false"`
	NetworkInterface  string `flag:"network-interface" default:"eth0"`
}

func genJobID() string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("hello world %d", time.Now().Unix())))
	byt := h.Sum(nil)
	he := hex.EncodeToString(byt)
	return he[0:10]
}

func (j *JobConfig) RegisterFlags(flag *flag.FlagSet) {
	structflag.RegisterFlags(j, flag)
	structflag.RegisterFlags(&j.Dataset, flag)
	j.Cluster.RegisterFlags(flag)
	j.ID = genJobID()
}

func (j *JobConfig) ParseParaConfig() {
	var err error
	j.ParaConfigs, err = para_config.LoadFile(j.ParaConfigFile)
	if err != nil {
		log.Panicf("%s: %v", `ParseParaConfig`, err)
	}
	for i, size := range j.ParaConfigs.Sizes() {
		log.Printf("ParaConfig[%d/%d]: %s", i+1, len(j.ParaConfigs), j.ParaConfigs[size])
	}
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

func (j *JobConfig) DistFlags(c TrainingConfig, rank int) []string {
	size := c.MDP.DPSize * c.MDP.PPSize * c.MDP.MPSize
	numProc := c.GPUPerNode
	if size < numProc { // min
		numProc = size
	}
	return []string{
		`--nproc_per_node`, str(numProc),
		`--nnodes`, str(c.NumNodes),
		`--node_rank`, str(rank),
		`--master_addr`, j.Cluster.Hosts[0],
		`--master_port`, `6000`,
	}
}

func (j *JobConfig) LogFlags(c TrainingConfig) []string {
	return []string{
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters`, `10`,
	}
}

func (j *JobConfig) TenplexFlags(c TrainingConfig, host string) []string {
	if j.NoTenplex {
		return nil
	}
	var cmd []string
	cmd = append(cmd, `--tenplex`)
	cmd = append(cmd, `--mlfs-path`, `/data/mlfs`)
	cmd = append(cmd, `--jobid`, j.ID)
	cmd = append(cmd, `--host-ip`, host)
	cmd = append(cmd, `--mlfs-port`, str(j.MLFSPort))
	if j.GenParaConfig {
		cmd = append(cmd, `--gen-para-config`)
	}
	return cmd
}

func (j *JobConfig) OtherFlags(c TrainingConfig) []string {
	const checkpoint_path = `/data/ckpt`
	var cmd []string
	args := []string{
		`--log-validation-ppl-to-tensorboard`,
		`--save`, checkpoint_path,
		`--load`, checkpoint_path,
		`--tensor-model-parallel-size`, str(c.MDP.MPSize),
		`--pipeline-model-parallel-size`, str(c.MDP.PPSize),
		`--tensorboard-dir`, path.Join(checkpoint_path, `tensorboard`),
		`--seed`, str(j.Seed),
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

func (j *JobConfig) Validate() {
	for _, scalingPoint := range j.Schedule {
		if _, ok := j.ParaConfigs[scalingPoint.Size]; !ok {
			log.Printf("schedule not defined for size %d", scalingPoint.Size)
		}
	}
}
