#!/bin/bash
set -e

export PYTHONPATH="/home/marcel/Elasticity/Repo/Megatron-LM"

python load.py \
    --mlfs-path /data/marcel/mlfs
