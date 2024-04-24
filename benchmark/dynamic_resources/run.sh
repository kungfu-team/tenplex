#!/bin/bash
set -e

. $(pwd)/../common.sh

hosts() {
    cat $(pwd)/hosts.txt | grep -v '#'
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
    echo -hosts "$(join $(hosts))"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
    echo -time-based
    echo -detect-self-ip ib0
    echo -seed 1234
    echo -no-shuffle

    echo -schedule-file "$(dirname $0)/schedule.json"
}

tenplex_flags() {
    common_flags
    echo -jobid dyn-res-ten
    echo -para-config "$(dirname $0)/tenplex-para-config.json"
}

tenplex_dp_flags() {
    common_flags
    echo -jobid dyn-res-tdp
    echo -para-config "$(dirname $0)/pytorch-para-config.json"
}

pytorch_flags() {
    common_flags
    echo -jobid dyn-res-pyt
    echo -para-config "$(dirname $0)/pytorch-para-config.json"
    echo -no-tenplex
}

tenplex-run $(tenplex_flags) >tenplex-dyn-res.log 2>&1

# tenplex-run $(tenplex_dp_flags) >tenplex-dp-dyn-res.log 2>&1

# sudo rm -fr /mnt/k1d2/ckpt/*
# tenplex-run $(pytorch_flags) >pytorch-dyn-res.log 2>&1
