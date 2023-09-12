#!/bin/bash

set -e

export PYTHONPATH="$HOME/Elasticity/Repo/Megatron-LM"

python test_load.py \
    --device-rank 0 \
    --mlfs-path "/mnt/mlfs/job/job-single"
