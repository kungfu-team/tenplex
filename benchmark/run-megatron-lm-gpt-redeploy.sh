#!/bin/bash
set -e

make

join_() {
    local IFS=$1
    shift
    echo "$*"
}

join() { join_ , $@; }

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    echo "10.10.10.3"
    echo "10.10.10.4"
}

./bin/tenplex-run \
    -image "kungfu.azurecr.io/mw-megatron-lm-update:latest" \
    -framework "megatron-lm" \
    -model "gpt" \
    -model-size "large" \
    -dataset "enwiki" \
    -batch-size 128 \
    -micro-batch-size 8 \
    -precision "fp16" \
    -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt" \
    -hosts "$(join $(list_hosts))" \
    -tenplex-prefix "$HOME/.tenplex" \
    -schedule-file "$(pwd)/schedule.json" \
    -mlfs-port 20010 \
    -gpu-per-host 4 \
    -gpu-per-container 4 \
    -user $USER \
    -redeploy true \
    -seq-length 1024 > tenplex-run.log 2>&1
