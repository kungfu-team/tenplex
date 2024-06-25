#!/bin/bash
set -e

make binaries

sudo rm -rf ~/.tenplex/training/*

./bin/tenplex-debug \
    -image "kungfu.azurecr.io/mw-megatron-lm-23.06-debug:latest" \
    -user $USER \
    -mlfs-port 20010 \
    -tenplex-prefix "$HOME/.tenplex" \
    -framework "megatron-lm" \
    -model "gpt" \
    -model-size "large" \
    -dataset "enwiki" \
    -batch-size 128 \
    -micro-batch-size 8 \
    -precision "fp16" \
    -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt" \
    -hosts "10.10.10.2" \
    -schedule-file "./benchmark/schedule.json" \
    -para-config "./benchmark/para-config.json" \
    -gpu-per-host 1 \
    -gpu-per-container 1 \
    -seq-length 1024 \
    -no-tenplex
