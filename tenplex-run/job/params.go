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
