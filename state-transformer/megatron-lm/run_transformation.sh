#!/bin/bash
set -e

export PYTHONPATH="/home/marcel/Elasticity/Repo/Megatron-LM"

python transformation.py \
    --input-dir /data/${USER}/united \
    --output-dir /data/${USER}/transformed \
    --ckpt-struct-dir ${HOME}/Elasticity/Repo/transformer-checkpoint/megatron-lm \
    --source-pp-degree 2 \
    --target-pp-degree 1 \
    --source-mp-degree 2 \
    --target-mp-degree 4 \
    --source-size 4 \
    --target-size 4
