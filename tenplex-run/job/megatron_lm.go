package job

import "github.com/kungfu-team/tenplex/tenplex-run/structflag"

type MegatronLM struct {
	SeqLength             int     `flag:"seq-length"`
	MaxPositionEmbeddings int     `flag:"max-position-embeddings"`
	LR                    float64 `flag:"lr"`
	MinLR                 float64 `flag:"min-lr"`
	LRWarmupFraction      float64 `flag:"lr-warmup-fraction"`
	LRDecayIters          int     `flag:"lr-decay-iters"`
	TrainIters            int     `flag:"train-iters"`
	MicroBatchSize        int     `flag:"micro-batch-size"`
	GlobalBatchSize       int     `flag:"global-batch-size"`
	VocabFile             string  `flag:"vocab-file"`
	Split                 string  `flag:"split"`
	DataPath              string  `flag:"data-path"`
	DistributedBackend    string  `flag:"distributed-backend"`
}

func (s MegatronLM) ToPyArgs() []string {
	return structflag.ToArgs(&s, structflag.LongFlagName)
}
