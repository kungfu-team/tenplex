#!/bin/bash
set -e

. $(dirname $0)/common.sh

list_hosts() {
    echo "10.10.10.2"
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
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -schedule-file "$(dirname $0)/schedule.json"
    echo -para-config "$(dirname $0)/para-config.json"
    echo -gpu-per-host 1
    echo -gpu-per-container 1
    echo -seq-length 1024
}

tenplex_run_with flags
