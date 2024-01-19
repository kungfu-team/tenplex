#!/bin/bash
set -e

make

    # -image "kungfu.azurecr.io/mw-megatron-lm-no-scheduler:latest" \
    # -image "kungfu.azurecr.io/mw-megatron-lm-no-scheduler-gpt-torch-save:latest" \
    # -index-url "https://tenplex.blob.core.windows.net/tenplexcontainer/gpt_enwiki_indices.txt" \
./bin/tenplex-run \
    -image "kungfu.azurecr.io/mw-megatron-lm-no-scheduler:latest" \
    -framework "megatron-lm" \
    -model "bert" \
    -model-size "large" \
    -dataset "openwebtext" \
    -batch-size 128 \
    -micro-batch-size 8 \
    -precision "fp16" \
    -hosts "10.10.10.1" \
    -index-url "/data/megatron-lm/bert/openwebtext/npzs_seq1024/indices.txt" \
    -tenplex-prefix "$HOME/.tenplex" \
    -schedule-file "$(pwd)/schedule.json" \
    -mlfs-port 20010 \
    -gpu-per-host 4 \
    -gpu-per-container 1 \
    -user marcel \
    -seq-length 1024 > tenplex-run.log 2>&1
