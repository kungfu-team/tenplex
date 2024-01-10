#!/bin/bash
set -e

make

./bin/tests \
    --ckpt-dir "/data/$USER/mlfs" \
    --ckpt-struct-dir "$HOME/Elasticity/Repo/tenplex-run/transformer-checkpoint" \
    --source-pp-degree 2 \
    --target-pp-degree 3 \
    --source-mp-degree 2 \
    --target-mp-degree 4 \
    --source-size 8 \
    --target-size 12 \
    --precision "fp16" \
    --input-timestamp "a" \
    --output-timestamp "b" \
    --hosts "10.10.10.1" \
    --mdp-library "megatron-lm" \
    --sequence-length 1024 \
    --target-rank 0
