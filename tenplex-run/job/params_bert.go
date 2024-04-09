package job

import "log"

func GenMegatronLMBERTCmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	cmd = append(cmd, jConf.DistFlags(c, rank)...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_bert.py`)
	var bert_args []string
	var bertSizeArgs = map[string]TransformerSize{
		`base`:  TFSize(12, 768, 12),
		`large`: TFSize(24, 1024, 16),
	}
	if ts, ok := bertSizeArgs[jConf.ModelSize]; ok {
		bert_args = append(bert_args, ts.ToPyArgs()...)
	} else {
		log.Fatalf("Model size not matching %s", jConf.ModelSize)
	}
	bert_args = append(bert_args, []string{
		`--seq-length`, `1024`, // default: 512
		`--max-position-embeddings`, `1024`, // default: 512
		`--lr`, `0.0001`,
		`--lr-decay-iters`, `10000`,
		`--train-iters`, str(c.TrainIters),
		`--min-lr`, `0.00001`,
		`--lr-warmup-fraction`, `0.01`,
		`--micro-batch-size`, str(jConf.MicroBatchSize), // default: 4
		`--global-batch-size`, str(jConf.BatchSize), // default: 32
		`--vocab-file`, `/workspace/Megatron-LM/vocab/bert-large-uncased-vocab.txt`,
		`--split`, `949,50,1`,
		`--data-path`, `/data/dataset/bert_text_sentence`,
		`--distributed-backend`, `nccl`,
	}...)
	cmd = append(cmd, bert_args...)
	cmd = append(cmd, jConf.LogFlags(c)...)
	cmd = append(cmd, jConf.TenplexFlags(c, host)...)
	cmd = append(cmd, jConf.OtherFlags(c)...)
	return cmd
}
