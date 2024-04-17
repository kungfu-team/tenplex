#!/bin/bash
set -e

. ./common.sh

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    echo "10.10.10.3"
    echo "10.10.10.4"
}

flags() {
    echo -image "kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest"
    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "large"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -tenplex-prefix "$HOME/.tenplex"
    echo -schedule-file "$(dirname $0)/schedule.json"
    echo -para-config "$(dirname $0)/para-config.json"
    echo -mlfs-port 20010
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -user $USER
    echo -seq-length 1024
}

tenplex-run $(flags) > gpt.log 2>&1

