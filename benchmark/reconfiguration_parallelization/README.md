# Reconfiguration Parallelizations
_Reconfiguration time with different parallelizations_

We examine the impact of the parallelization configuration on reconfiguration time for different model sizes. We deploy Tenplex and Tenplex-Central, which manages the state in a single node, with the different GPT-3 models on the on-premise cluster. For data parallelism (D), we change the configuration from (M, P, D) = (4, 2, 1) to (4, 2, 2); for pipeline parallelism (P) from (4, 2, 1) to (4, 4, 1); and for model parallelism (M) from (4, 2, 1) to (8, 2, 1).
