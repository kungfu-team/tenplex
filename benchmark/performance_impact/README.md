# Performance impact
_Fig. 3. Performance impact of different parallelization configurations on 16 GPUs_

When the GPU resources of a DL job change at runtime, a parallelization configuration that was optimal at deployment time may no longer be optimal with the new GPUs. We demonstrate this empirically in Fig. 3, which shows the training throughput (in samples/second) when training BERT and GPT-3 models using Megatron-LM on 16 GPUs under a range of parallelization configurations. Each parallelization configuration varies the degree of model, pipeline and data parallelism, and thus alters the GPU allocation.
