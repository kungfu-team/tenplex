#!/bin/bash
set -e

. $(pwd)/../common.sh

    # echo -plan ./tenplex-schedule-test.json
    # echo -plan ./pytorch-schedule-test.json

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    # echo "10.10.10.3"
    # echo "10.10.10.4"
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
    echo -schedule-file "$(pwd)/tenplex-schedule-test.json"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
    echo -time-based
    echo -scheduler-ip "http://10.10.10.1:22222"
}

tenplex_flags() {
    echo -schedule-file "$(pwd)/tenplex-schedule-test.json"
}

pytorch_flags() {
    echo -schedule-file "$(pwd)/pytorch-schedule-test.json"
}

tenplex-run $(common_flags) $(tenplex_flags) > dyn.log 2>&1
