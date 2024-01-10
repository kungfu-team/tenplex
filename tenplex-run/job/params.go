package job

import (
	"fmt"
	"path"
	"strconv"
)

var str = strconv.Itoa

type MDPConfig struct {
	NumNodes             int
	GPUPerNode           int
	ModelParallelSize    int
	PipelineParallelSize int

	TrainIters int

	LogInterval  int
	SaveInterval int
	EvalInterval int

	Precision string
}

func GenMegatronLMBERTCmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	distributed_args := []string{
		`--nproc_per_node`, str(c.GPUPerNode),
		`--nnodes`, str(c.NumNodes),
		`--node_rank`, str(rank),
		// `--master_addr`, fmt.Sprintf("trainer-%s-00", jobID),
		`--master_addr`, jConf.Cluster.Hosts[0],
		`--master_port`, `6000`,
	}
	cmd = append(cmd, distributed_args...)
	cmd = append(cmd, `/workspace/Megatron-LM/pretrain_bert.py`)
	bert_args := []string{}
	if jConf.ModelSize == "base" {
		bert_args = append(bert_args,
			`--num-layers`, `12`,
			`--hidden-size`, `768`,
			`--num-attention-heads`, `12`,
		)
	} else if jConf.ModelSize == "large" {
		bert_args = append(bert_args,
			`--num-layers`, `24`,
			`--hidden-size`, `1024`,
			`--num-attention-heads`, `16`,
		)
	} else {
		panic(fmt.Sprintf("Model size not matching %v", jConf.ModelSize))
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
	}...)
	if c.Precision == "fp16" {
		bert_args = append(bert_args,
			`--fp16`)
	}
	cmd = append(cmd, bert_args...)
	output_args := []string{
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters`, `0`, // default: 10
	}
	cmd = append(cmd, output_args...)
	checkpoint_path := `/data/ckpt`
	args := []string{
		`--save`, checkpoint_path,
		`--load`, checkpoint_path,
		`--data-path`, `/data/dataset/bert_text_sentence`,
		`--tensor-model-parallel-size`, str(c.ModelParallelSize),
		`--pipeline-model-parallel-size`, str(c.PipelineParallelSize),
	}
	cmd = append(cmd, args...)
	cmd = append(cmd, `--tensorboard-dir`, path.Join(checkpoint_path, `tensorboard`))
	cmd = append(cmd, `--mlfs-path`, `/data/mlfs`)
	cmd = append(cmd, `--jobid`, jobID)
	cmd = append(cmd, `--host-ip`, host)
	cmd = append(cmd, `--mlfs-port`, str(jConf.MLFSPort))
	if jConf.SchedulerIP != "" {
		cmd = append(cmd, `--scheduler-addr`, jConf.SchedulerIP)
	}
	return cmd
}

func GenMegatronLMGPTCmd(c MDPConfig, rank int, jobID string, host string, jConf *JobConfig, masterAddr string) []string {
	cmd := []string{
		`torchrun`,
	}
	distributed_args := []string{
		`--nproc_per_node`, str(c.GPUPerNode),
		`--nnodes`, str(c.NumNodes),
		`--node_rank`, str(rank),
		// `--master_addr`, fmt.Sprintf("trainer-%s-00", jobID),
		`--master_addr`, masterAddr,
		`--master_port`, `6000`,
	}
	cmd = append(cmd, distributed_args...)
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

	if c.Precision == "fp16" {
		gpt_args = append(gpt_args,
			`--fp16`)
	}
	cmd = append(cmd, gpt_args...)
	output_args := []string{
		`--log-interval`, str(c.LogInterval),
		`--save-interval`, str(c.SaveInterval),
		`--eval-interval`, str(c.EvalInterval),
		`--eval-iters`, `0`, // default: 10
	}
	cmd = append(cmd, output_args...)
	checkpoint_path := `/data/ckpt`
	args := []string{
		`--save`, checkpoint_path,
		`--load`, checkpoint_path,
		`--data-path`, `/data/dataset/???`,
		`--tensor-model-parallel-size`, str(c.ModelParallelSize),
		`--pipeline-model-parallel-size`, str(c.PipelineParallelSize),
		// `--DDP-impl`, `local`,
		`--distributed-backend`, `nccl`,
	}
	cmd = append(cmd, args...)
	cmd = append(cmd, `--tensorboard-dir`, path.Join(checkpoint_path, `tensorboard`))
	cmd = append(cmd, `--mlfs-path`, `/data/mlfs`)
	cmd = append(cmd, `--jobid`, jobID)
	cmd = append(cmd, `--host-ip`, host)
	cmd = append(cmd, `--mlfs-port`, str(jConf.MLFSPort))
	if jConf.SchedulerIP != "" {
		cmd = append(cmd, `--scheduler-addr`, jConf.SchedulerIP)
	}
	return cmd
}

func GenDeepspeedCommandOld(c MDPConfig, jConf *JobConfig) []string {
	cmd := []string{
		`deepspeed`,
		`--hostfile=/data/ckpt/hostfile.txt`,
		`--num_nodes`, str(c.NumNodes),
		`--num_gpus`, str(c.GPUPerNode),
		`pretrain_gpt2.py`,
		`--model-parallel-size`, str(c.ModelParallelSize),
		`--pipe-parallel-size`, str(c.PipelineParallelSize),
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
func GenDeepspeedCommand(c MDPConfig, jConf *JobConfig) []string {
	cmd := []string{
		`deepspeed`,
		`--hostfile=/data/ckpt/hostfile.txt`,
		`--num_nodes`, str(c.NumNodes),
		`--num_gpus`, str(c.GPUPerNode),
		`pretrain_gpt.py`,
		`--tensor-model-parallel-size`, str(c.ModelParallelSize),
		`--pipeline-model-parallel-size`, str(c.PipelineParallelSize),
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

func GenMegatronDeepspeedCommand(c MDPConfig, rank int, jobID string, jConf *JobConfig) []string {
	cmd := []string{
		`torchrun`,
	}
	cmd = append(cmd, []string{
		`--nproc_per_node`, str(c.GPUPerNode),
		`--nnodes`, str(c.NumNodes),
		`--node_rank`, str(rank),
		`--master_addr`, fmt.Sprintf("trainer-%s-00", jobID),
		`--master_port`, `6000`,
	}...)
	cmd = append(cmd, `/workspace/Megatron-DeepSpeed/pretrain_gpt.py`)
	cmd = append(cmd, []string{
		`--tensor-model-parallel-size`, str(c.ModelParallelSize),
		`--pipeline-model-parallel-size`, str(c.PipelineParallelSize),
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
