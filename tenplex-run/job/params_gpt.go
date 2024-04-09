package job

import "log"

func GenMegatronLMGPTCmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	cmd = append(cmd, jConf.DistFlags(c, rank)...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_gpt.py`)
	gpt_args := []string{
		`--micro-batch-size`, str(jConf.MicroBatchSize),
		`--global-batch-size`, str(jConf.BatchSize),
		`--seq-length`, `1024`,
		`--max-position-embeddings`, `1024`,
		`--train-iters`, str(c.TrainIters),
		`--lr-decay-iters`, `10000`,
		`--lr-warmup-fraction`, `0.01`,
		`--vocab-file`, `/workspace/Megatron-LM/vocab/gpt2-vocab.json`,
		`--merge-file`, `/workspace/Megatron-LM/vocab/gpt2-merges.txt`,
		`--data-impl`, `mmap`,
		`--split`, `949,50,1`,
		`--lr`, `0.00015`,
		`--lr-decay-style`, `cosine`,
		`--min-lr`, `1.0e-5`,
		`--weight-decay`, `1e-2`,
		`--clip-grad`, `1.0`,
		`--lr-warmup-fraction`, `.01`,
	}
	var gptSizeArgs = map[string]TransformerSize{
		`medium`: TFSize(24, 1024, 16),
		`large`:  TFSize(24, 1536, 16),
		`xl`:     TFSize(24, 2064, 24), // should be 2048 but hidden_size % num_attention_heads == 0
		`2.7B`:   TFSize(32, 2560, 32),
		`6.7B`:   TFSize(32, 4096, 32),
	}
	if ts, ok := gptSizeArgs[jConf.ModelSize]; ok {
		gpt_args = append(gpt_args, ts.ToPyArgs()...)
	} else {
		log.Fatalf("Model size not matching %s", jConf.ModelSize)
	}
	cmd = append(cmd, gpt_args...)
	args := []string{
		`--data-path`, `/data/dataset/gpt-2/my-gpt2_text_document`,
		// `--DDP-impl`, `local`,
		`--distributed-backend`, `nccl`,
	}
	cmd = append(cmd, args...)
	cmd = append(cmd, jConf.LogFlags(c)...)
	cmd = append(cmd, jConf.TenplexFlags(c, host)...)
	cmd = append(cmd, jConf.OtherFlags(c)...)
	return cmd
}
