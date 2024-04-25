package job

import (
	"strconv"

	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

var str = strconv.Itoa

type MDPConfig struct {
	NumNodes             int
	GPUPerNode           int
	ModelParallelSize    int
	PipelineParallelSize int

	TrainIters int

	LogInterval  int
	SaveInterval int
	EvalInterval int

	Precision string
}

type GenCmdFunc func(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string

type TransformerSize struct {
	Layers         int `flag:"num-layers"`
	HiddenSize     int `flag:"hidden-size"`
	AttentionHeads int `flag:"num-attention-heads"`
}

func (s TransformerSize) ToPyArgs() []string {
	return structflag.ToArgs(&s, structflag.LongFlagName)
}

func TFSize(nl, hs, ah int) TransformerSize {
	return TransformerSize{
		Layers:         nl,
		HiddenSize:     hs,
		AttentionHeads: ah,
	}
}
