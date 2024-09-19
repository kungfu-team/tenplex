# Reconfiguration Horovod
_Fig. 13. Reconfiguration time against Horovod_

We also compare Tenplex’s overhead to Horovod, a distributed training library without elasticity support, and Horovod-Elastic, which also supports scaling under data parallelism only by periodically checkpointing the model state. We deploy a ResNet50 model with the ImageNet dataset in the on-premise cluster, and measure throughput when training on 2 GPUs.

Note: The absolute number of throughput depends on a number of factors, including GPU generation and disk speed. We expect the throughput of tenplex is at the same level of Horovod, and therefore faster than Horovod-Elastic.
