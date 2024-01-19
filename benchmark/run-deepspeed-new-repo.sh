#!/bin/bash
set -e

make

    # -framework "deepspeed-new-repo" \
./bin/tenplex-run \
    -image "kungfu.azurecr.io/mw-megatron-deepspeed:latest" \
    -framework "deepspeed" \
    -model "gpt" \
    -model-size "medium" \
    -dataset "enwiki" \
    -batch-size 128 \
    -micro-batch-size 8 \
    -precision "fp16" \
    -hosts "10.10.10.1" \
    -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt" \
    -tenplex-prefix "$HOME/.tenplex" \
    -scheduler-ip "10.10.10.1" \
    -schedule-file "$(pwd)/schedule.json" \
    -mlfs-port 20010 \
    -gpu-per-host 4 \
    -gpu-per-container 4 \
    -user marcel \
    -seq-length 1024 > tenplex-run.log 2>&1
