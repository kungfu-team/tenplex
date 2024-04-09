package job

import "fmt"

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

	if jConf.ModelSize == "medium" {
		gpt_args = append(gpt_args,
			`--num-layers`, `24`,
			`--hidden-size`, `1024`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "large" {
		gpt_args = append(gpt_args,
			`--num-layers`, `24`,
			`--hidden-size`, `1536`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "xl" {
		gpt_args = append(gpt_args,
			`--num-layers`, `24`,
			`--hidden-size`, `2064`, // should be 2048 but hidden_size % num_attention_heads == 0
			`--num-attention-heads`, `24`,
		)
	} else if jConf.ModelSize == "2.7B" {
		gpt_args = append(gpt_args,
			`--num-layers`, `32`,
			`--hidden-size`, `2560`,
			`--num-attention-heads`, `32`,
		)
	} else if jConf.ModelSize == "6.7B" {
		gpt_args = append(gpt_args,
			`--num-layers`, `32`,
			`--hidden-size`, `4096`,
			`--num-attention-heads`, `32`,
		)
	} else {
		panic(fmt.Sprintf("Model size not matching %v", jConf.ModelSize))
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
