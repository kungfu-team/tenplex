#!/bin/sh

set -e

. $(dirname $0)/common.sh

hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    echo "10.10.10.3"
    echo "10.10.10.4"
}

model_sizes() {
    echo "6.7B"
    echo "2.7B"
    echo "xl"
}

comb_flags() {
    echo -model "gpt"
    echo -dataset "enwiki"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt"
    echo -hosts $(join $(hosts))
    echo -schedule "schedule.json"
    echo -model-sizes $(join $(model_sizes))
    echo -batch-sizes 128
    echo -micro-batch-sizes 8
    echo -para-config "para-config.json"
    echo -redeploy
}

tenplex-multi-experiment 2>&1 | tee redeploy.log
tenplex-multi-experiment -central 2>&1 | tee redeploy_central.log
