package job

import "log"

func GenMegatronLMGPTCmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	cmd = append(cmd, jConf.DistFlags(c, rank)...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_gpt.py`)
	mlm := MegatronLM{
		MicroBatchSize:        jConf.MicroBatchSize,
		GlobalBatchSize:       jConf.BatchSize,
		SeqLength:             1024,
		MaxPositionEmbeddings: 1024,
		TrainIters:            c.TrainIters,
		LRDecayIters:          10000,
		LRWarmupFraction:      0.01,
		VocabFile:             `/workspace/Megatron-LM/vocab/gpt2-vocab.json`,
		Split:                 `949,50,1`,
		LR:                    0.00015,
		MinLR:                 0.00001,
		DataPath:              `/data/dataset/gpt-2/my-gpt2_text_document`,
		DistributedBackend:    `nccl`,
	}
	gpt_args := mlm.ToPyArgs()
	gpt_args = append(gpt_args,
		`--merge-file`, `/workspace/Megatron-LM/vocab/gpt2-merges.txt`,
		`--data-impl`, `mmap`,
		`--lr-decay-style`, `cosine`,
		`--weight-decay`, `1e-2`,
		`--clip-grad`, `1.0`,
	)
	var sizes = map[string]TransformerSize{
		`medium`: TFSize(24, 1024, 16),
		`large`:  TFSize(24, 1536, 16),
		`xl`:     TFSize(24, 2064, 24), // should be 2048 but hidden_size % num_attention_heads == 0
		`2.7B`:   TFSize(32, 2560, 32),
		`6.7B`:   TFSize(32, 4096, 32),
	}
	if ts, ok := sizes[jConf.ModelSize]; ok {
		gpt_args = append(gpt_args, ts.ToPyArgs()...)
	} else {
		log.Fatalf("Model size not matching %s", jConf.ModelSize)
	}
	cmd = append(cmd, gpt_args...)
	cmd = append(cmd, jConf.LogFlags(c)...)
	cmd = append(cmd, jConf.TenplexFlags(c, host)...)
	cmd = append(cmd, jConf.OtherFlags(c)...)
	return cmd
}
