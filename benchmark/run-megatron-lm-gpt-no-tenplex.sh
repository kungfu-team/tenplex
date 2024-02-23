#!/bin/bash
set -e

. $(dirname $0)/common.sh

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.3"
}

flags() {
    base_flags

    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "large"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -schedule-file "$(pwd)/schedule.json"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
    echo -no-tenplex
}

tenplex_run_with flags
