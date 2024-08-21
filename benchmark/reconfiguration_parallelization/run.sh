#!/bin/sh

set -e

. $(dirname $0)/../common.sh

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
    base_flags
    echo -model "gpt"
    echo -dataset "enwiki"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt"
    echo -hosts $(join $(hosts))
    echo -schedule "$(dirname $0)/schedule.json"
    echo -model-sizes $(join $(model_sizes))
    echo -batch-sizes 128
    echo -micro-batch-sizes 8
    echo -central
}

tenplex-multi-experiment $(comb_flags) -para-config para-config-dp.json 2>&1 | tee para_dp.log
tenplex-multi-experiment $(comb_flags) -para-config para-config-tp.json 2>&1 | tee para_tp.log
tenplex-multi-experiment $(comb_flags) -para-config para-config-pp.json 2>&1 | tee para_pp.log

python plot.py
