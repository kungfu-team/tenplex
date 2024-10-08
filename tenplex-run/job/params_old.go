package job

import (
	"fmt"
)

func GenDeepspeedCommandOld(c TrainingConfig, jConf *JobConfig) []string {
	cmd := []string{
		`deepspeed`,
		`--hostfile=/data/ckpt/hostfile.txt`,
		`--num_nodes`, str(c.NumNodes),
		`--num_gpus`, str(c.GPUPerNode),
		`pretrain_gpt2.py`,
		`--model-parallel-size`, str(c.MDP.MPSize),
		`--pipe-parallel-size`, str(c.MDP.PPSize),
	}

	if jConf.ModelSize == "medium" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `1024`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "large" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `1536`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "xl" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `2064`, // should be 2048 but hidden_size % num_attention_heads == 0
			`--num-attention-heads`, `24`,
		)
	} else if jConf.ModelSize == "2.7B" {
		cmd = append(cmd,
			`--num-layers`, `32`,
			`--hidden-size`, `2560`,
			`--num-attention-heads`, `32`,
		)
	} else if jConf.ModelSize == "6.7B" {
		cmd = append(cmd,
			`--num-layers`, `32`,
			`--hidden-size`, `4096`,
			`--num-attention-heads`, `32`,
		)
	} else {
		panic(fmt.Sprintf("Model size not matching %v", jConf.ModelSize))
	}

	cmd = append(cmd, []string{
		`--seq-length`, `1024`,
		`--max-position-embeddings 1024`,
		`--batch-size`, str(jConf.BatchSize),
		`--gas`, `1`,
		`--train-iters`, str(c.TrainIters),
		`--lr-decay-iters`, `10000`,
		`--save`, `/data/ckpt`,
		`--load`, `/data/ckpt`,
		`--data-path`, `/data/dataset/my-gpt2_text_document`,
		`--vocab-file`, `/data/dataset/gpt2-vocab.json`,
		`--merge-file`, `/data/dataset/gpt2-merges.txt`,
		`--data-impl mmap`,
		`--split 949,50,1`,
		`--distributed-backend nccl`,
		`--lr 1.5e-4`,
		`--lr-decay-style cosine`,
		`--min-lr 1.0e-5`,
		`--weight-decay 1e-2`,
		`--clip-grad 1.0`,
		`--warmup 0.01`,
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters 10`,
		`--tensorboard-dir`, `/data/ckpt/tensorboard`,
		`--deepspeed`,
		`--deepspeed_config`, `examples/ds_config.json`,
	}...)
	return cmd
}

func GenDeepspeedCommand(c TrainingConfig, jConf *JobConfig) []string {
	cmd := []string{
		`deepspeed`,
		`--hostfile=/data/ckpt/hostfile.txt`,
		`--num_nodes`, str(c.NumNodes),
		`--num_gpus`, str(c.GPUPerNode),
		`pretrain_gpt.py`,
		`--tensor-model-parallel-size`, str(c.MDP.MPSize),
		`--pipeline-model-parallel-size`, str(c.MDP.PPSize),
	}

	if jConf.ModelSize == "medium" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `1024`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "large" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `1536`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "xl" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `2064`, // should be 2048 but hidden_size % num_attention_heads == 0
			`--num-attention-heads`, `24`,
		)
	} else if jConf.ModelSize == "2.7B" {
		cmd = append(cmd,
			`--num-layers`, `32`,
			`--hidden-size`, `2560`,
			`--num-attention-heads`, `32`,
		)
	} else if jConf.ModelSize == "6.7B" {
		cmd = append(cmd,
			`--num-layers`, `32`,
			`--hidden-size`, `4096`,
			`--num-attention-heads`, `32`,
		)
	} else {
		panic(fmt.Sprintf("Model size not matching %v", jConf.ModelSize))
	}

	cmd = append(cmd, []string{
		`--seq-length`, `1024`,
		`--max-position-embeddings 1024`,
		`--micro-batch-size`, str(jConf.MicroBatchSize),
		`--global-batch-size`, str(jConf.BatchSize),
		`--train-iters`, str(c.TrainIters),
		`--lr-decay-iters`, `10000`,
		`--save`, `/data/ckpt`,
		`--load`, `/data/ckpt`,
		`--data-path`, `/data/dataset/my-gpt2_text_document`,
		`--vocab-file`, `/data/dataset/gpt2-vocab.json`,
		`--merge-file`, `/data/dataset/gpt2-merges.txt`,
		`--data-impl mmap`,
		`--split 949,50,1`,
		`--distributed-backend nccl`,
		`--lr 1.5e-4`,
		`--lr-decay-style cosine`,
		`--min-lr 1.0e-5`,
		`--weight-decay 1e-2`,
		`--clip-grad 1.0`,
		`--lr-warmup-fraction`, `0.01`,
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters 10`,
		`--tensorboard-dir`, `/data/ckpt/tensorboard`,
		`--deepspeed`,
		`--deepspeed_config`, `examples/ds_config.json`,
	}...)
	return cmd
}

func GenMegatronDeepspeedCommand(c TrainingConfig, rank int, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	cmd = append(cmd, []string{
		`--nproc_per_node`, str(c.GPUPerNode),
		`--nnodes`, str(c.NumNodes),
		`--node_rank`, str(rank),
		`--master_addr`, fmt.Sprintf("trainer-%s-00", jConf.ID),
		`--master_port`, `6000`,
	}...)
	cmd = append(cmd, `/workspace/Megatron-DeepSpeed/pretrain_gpt.py`)
	cmd = append(cmd, []string{
		`--tensor-model-parallel-size`, str(c.MDP.MPSize),
		`--pipeline-model-parallel-size`, str(c.MDP.PPSize),
	}...)
	if jConf.ModelSize == "medium" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `1024`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "large" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `1536`,
			`--num-attention-heads`, `16`,
		)
	} else if jConf.ModelSize == "xl" {
		cmd = append(cmd,
			`--num-layers`, `24`,
			`--hidden-size`, `2064`, // should be 2048 but hidden_size % num_attention_heads == 0
			`--num-attention-heads`, `24`,
		)
	} else if jConf.ModelSize == "2.7B" {
		cmd = append(cmd,
			`--num-layers`, `32`,
			`--hidden-size`, `2560`,
			`--num-attention-heads`, `32`,
		)
	} else if jConf.ModelSize == "6.7B" {
		cmd = append(cmd,
			`--num-layers`, `32`,
			`--hidden-size`, `4096`,
			`--num-attention-heads`, `32`,
		)
	} else {
		panic(fmt.Sprintf("Model size not matching %v", jConf.ModelSize))
	}

	cmd = append(cmd, []string{
		`--seq-length`, `1024`,
		`--max-position-embeddings 1024`,
		`--micro-batch-size`, str(jConf.MicroBatchSize),
		`--global-batch-size`, str(jConf.BatchSize),
		`--train-iters`, str(c.TrainIters),
		`--lr-decay-iters`, `10000`,
		`--save`, `/data/ckpt`,
		`--load`, `/data/ckpt`,
		`--data-path`, `/data/dataset/my-gpt2_text_document`,
		`--vocab-file`, `/workspace/Megatron-DeepSpeed/vocab/gpt2-vocab.json`,
		`--merge-file`, `/workspace/Megatron-DeepSpeed/vocab/gpt2-merges.txt`,
		`--data-impl`, `mmap`,
		`--split 949,50,1`,
		`--distributed-backend nccl`,
		`--lr 1.5e-4`,
		`--lr-decay-style cosine`,
		`--min-lr 1.0e-5`,
		`--weight-decay 1e-2`,
		`--clip-grad 1.0`,
		`--lr-warmup-fraction`, `0.01`,
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters 10`,
		`--fp16`,
	}...)
	return cmd
}
