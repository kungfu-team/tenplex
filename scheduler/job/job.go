package job

import (
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/ds"
	"github.com/kungfu-team/tenplex/scheduler/scalepoint"
)

type Job struct {
	Framework      string
	Precision      string
	BatchSize      int
	MicroBatchSize int
	SequenceLength int
	Dataset        ds.Dataset
	Image          string
	Model          string
	ID             string
	Steps          int
	ModelSize      string
	NumLayers      int
	VocabSize      int
	Failure        int
}

func ShowJobIds(jss ...[]Job) string {
	var ids []string
	for _, js := range jss {
		for _, j := range js {
			ids = append(ids, j.ID)
		}
	}
	return strings.Join(ids, ",")
}

type TimedJob struct {
	Job    Job
	Timing []scalepoint.ScalePoint
}
