package job

import "strconv"

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
	NumLayers      int
	HiddenSize     int
	AttentionHeads int
}

func (s TransformerSize) ToPyArgs() []string {
	return []string{
		`--num-layers`, str(s.NumLayers),
		`--hidden-size`, str(s.HiddenSize),
		`--num-attention-heads`, str(s.AttentionHeads),
	}
}

func TFSize(nl, hs, ah int) TransformerSize {
	return TransformerSize{
		NumLayers:      nl,
		HiddenSize:     hs,
		AttentionHeads: ah,
	}
}
