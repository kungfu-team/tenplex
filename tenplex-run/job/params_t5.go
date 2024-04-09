package job

// https://github.com/kungfu-team/Megatron-LM/blob/mw/23.06/examples/pretrain_t5_distributed.sh
func GenMegatronLMT5Cmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	var cmd = []string{
		`torchrun`,
	}
	cmd = append(cmd, jConf.DistFlags(c, rank)...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_t5.py`)

	cmd = append(cmd, t5Args(c, jConf)...)
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

func t5Args(c MDPConfig, jConf *JobConfig) []string {
	t5_args := []string{
		`--micro-batch-size`, str(jConf.MicroBatchSize),
		`--global-batch-size`, str(jConf.BatchSize),
		`--seq-length`, str(1024),
		`--encoder-seq-length`, str(512),
		`--decoder-seq-length`, str(128),
		`--max-position-embeddings`, str(512),
		`--train-iters`, str(c.TrainIters),
		`--lr-decay-iters`, `10000`,
		`--lr-warmup-fraction`, `0.01`,
		// `--vocab-file`, `/workspace/Megatron-LM/vocab/gpt2-vocab.json`,
		`--vocab-file`, `/workspace/Megatron-LM/vocab/bert-large-uncased-vocab.txt`,
		`--data-impl`, `mmap`,
		`--split`, `949,50,1`,
		`--lr`, `0.00015`,
		`--lr-decay-style`, `cosine`,
		`--min-lr`, `1.0e-5`,
		`--weight-decay`, `1e-2`,
		`--clip-grad`, `1.0`,
		`--lr-warmup-fraction`, `.01`,
		`--kv-channels`, str(64),
		`--ffn-hidden-size`, str(3072),
		// `--pipeline-model-parallel-split-rank`
	}

	size_args := []string{
		`--num-layers`, str(12),
		`--hidden-size`, str(768),
		`--num-attention-heads`, str(12),
	}

	t5_args = append(t5_args, size_args...)
	return t5_args
}
