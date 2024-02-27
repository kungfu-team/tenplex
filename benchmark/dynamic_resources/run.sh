#!/bin/bash
set -e

. $(pwd)/../common.sh

list_hosts() {
    cat $(pwd)/../hosts.txt | head -n 2
}

common_flags() {
    base_flags

    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "xl"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
    echo -time-based
    echo -detect-self-ip ib0
}

tenplex_flags() {
    common_flags
    echo -schedule-file "$(pwd)/tenplex-schedule-test.json"
}

pytorch_flags() {
    common_flags
    echo -schedule-file "$(pwd)/pytorch-schedule-test.json"
    echo -no-tenplex
}

tenplex_run_with tenplex_flags

# sudo rm -fr /mnt/k1d2/ckpt/*
# tenplex_run_with pytorch_flags
