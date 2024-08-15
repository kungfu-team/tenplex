# Failure recovery
_Fig. 11. Failure recovery time (GPT-3 2.7 B)_

We explore how Tenplex manages to recover efficiently from failures, even in scenarios that require dynamic reconfiguration due to a change in the number of GPUs. We emulate faults of 4, 8, and 12 GPUs and measure the failure recovery and reconfiguration time. We use the GPT-3 2.7 B model with the Wikipedia dataset on the on-premise cluster. We compare Tenplex to a system that always recovers from the last checkpoint , which results in an average loss of 50 training steps. The parallelization configuration is (M, D, P) = (4, 2, 2), i.e. there are two model replicas.
