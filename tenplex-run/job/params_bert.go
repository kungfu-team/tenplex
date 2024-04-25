package job

import "log"

func GenMegatronLMBERTCmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	cmd = append(cmd, jConf.DistFlags(c, rank)...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_bert.py`)
	var sizes = map[string]TransformerSize{
		`base`:  TFSize(12, 768, 12),
		`large`: TFSize(24, 1024, 16),
	}
	mlm := MegatronLM{
		SeqLength:             1024, // default: 512
		MaxPositionEmbeddings: 1024, // default: 512
		LR:                    0.0001,
		LRDecayIters:          10000,
		TrainIters:            c.TrainIters,
		MinLR:                 0.00001,
		LRWarmupFraction:      0.01,
		MicroBatchSize:        jConf.MicroBatchSize,
		GlobalBatchSize:       jConf.BatchSize,
		VocabFile:             `/workspace/Megatron-LM/vocab/bert-large-uncased-vocab.txt`,
		Split:                 `949,50,1`,
		DataPath:              `/data/dataset/bert_text_sentence`,
		DistributedBackend:    `nccl`,
	}
	bert_args := mlm.ToPyArgs()
	if ts, ok := sizes[jConf.ModelSize]; ok {
		bert_args = append(bert_args, ts.ToPyArgs()...)
	} else {
		log.Fatalf("Model size not matching %s", jConf.ModelSize)
	}
	cmd = append(cmd, bert_args...)
	cmd = append(cmd, jConf.LogFlags(c)...)
	cmd = append(cmd, jConf.TenplexFlags(c, host)...)
	cmd = append(cmd, jConf.OtherFlags(c)...)
	return cmd
}
