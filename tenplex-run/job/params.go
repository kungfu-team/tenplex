package job

import (
	"strconv"

	"github.com/kungfu-team/tenplex/tenplex-run/para_config"
)

var str = strconv.Itoa

type TrainingConfig struct {
	NumNodes   int
	GPUPerNode int
	MDP        para_config.MultiDimensionalParallelism

	TrainIters int

	LogInterval  int
	SaveInterval int
	EvalInterval int

	Precision string
}

type GenCmdFunc func(c TrainingConfig, rank int, jobID string, host string, jConf *JobConfig) []string

type TransformerSize struct {
	Layers         int
	HiddenSize     int
	AttentionHeads int
}

func (s TransformerSize) ToPyArgs() []string {
	return []string{
		`--num-layers`, str(s.Layers),
		`--hidden-size`, str(s.HiddenSize),
		`--num-attention-heads`, str(s.AttentionHeads),
	}
}

func TFSize(nl, hs, ah int) TransformerSize {
	return TransformerSize{
		Layers:         nl,
		HiddenSize:     hs,
		AttentionHeads: ah,
	}
}
