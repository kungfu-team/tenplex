package job

// https://github.com/kungfu-team/Megatron-LM/blob/mw/23.06/examples/pretrain_t5_distributed.sh
func GenMegatronLMT5Cmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	var cmd = []string{
		`torchrun`,
	}
	cmd = append(cmd, jConf.DistFlags(c, rank)...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_t5.py`)

	cmd = append(cmd, t5Args(jConf)...)
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

func t5Args(jConf *JobConfig) []string {
	t5_args := []string{
		`--micro-batch-size`, str(jConf.MicroBatchSize),
		`--global-batch-size`, str(jConf.BatchSize),
	}

	return t5_args
}
