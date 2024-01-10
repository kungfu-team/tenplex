#!/bin/bash
set -e

for i in {0..7}
do
    python transformation.py \
        --input-dir /data/${USER}/united \
        --output-dir /data/${USER}/transformed \
        --ckpt-struct-dir ${HOME}/Elasticity/Repo/transformer-checkpoint/deepspeed \
        --source-pp-degree 2 \
        --target-pp-degree 4 \
        --source-mp-degree 2 \
        --target-mp-degree 2 \
        --source-size 8 \
        --target-size 8 \
        --target-rank ${i}
done
