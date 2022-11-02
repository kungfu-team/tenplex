#!/bin/bash
set -e

export PYTHONPATH="/home/marcel/Elasticity/Repo/Megatron-LM"

python save_all.py \
    --mlfs-path "/data/marcel/mlfs" \
    --ckpt-path "/data/marcel/training" \
    --step 50 \
    --size 4 \
    --timestamp "a"
