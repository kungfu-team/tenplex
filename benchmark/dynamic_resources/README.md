# Dynamic resources
_Fig. 9. Elastic DL job convergence with multi-dimensional parallelism under dynamic GPU changes_

First, we explore the benefits of supporting elasticity in DL jobs with multi-dimensional parallelism, scaling across all parallelism dimensions when the GPU allocation changes.
In this experiment, we train DL jobs with the GPT-3 XL model on the on-premise 16-GPU cluster. The job runtime and elastic scaling events are derived based on Microsoft’s Philly trace: over the runtime of 538 mins, we scale based on the average every 35 mins. During a scaling event, we change the number of GPUs for a job between 16, 8, and 4 GPUs.

# Run
```sh
./run.sh
```

## Note
The dynamic resources experiment runs for about 24 hours
