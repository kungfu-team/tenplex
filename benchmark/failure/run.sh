#!/bin/bash
set -e

hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    echo "10.10.10.3"
    echo "10.10.10.4"
}


flags() {
    echo -image "kungfu.azurecr.io/mw-megatron-lm-update:v0.0.1"
    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "2.7B"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt"
    echo -hosts "$(join $(hosts))"
    echo -tenplex-prefix "$HOME/.tenplex"
    echo -schedule-file "schedule.json"
    echo -para-config "para-config.json"
    echo -mlfs-port 20010
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -user $USER
    echo -seq-length 1024
}

for i in 4 8 12
do
    tenplex-run $(flags) -failure $i 2>&1 | tee failure_$i.log
    mv logs logs_$i
done

python plot.py
