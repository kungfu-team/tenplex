#!/bin/sh

set -e

. $(dirname $0)/common.sh

hosts() {
    echo "10.10.10.1"
    # echo "10.10.10.2"
    echo "10.10.10.3"
    # echo "10.10.10.4"
}

model_sizes() {
    # echo "6.7B"
    # echo "2.7B"
    # echo "xl"
    echo "large"
}

batch_sizes() {
    echo 128
    # echo 256
}

micro_batch_sizes() {
    echo 8
}

schedules() {
    # ls data/schedule-*.json | sort
    echo "$(dirname $0)/schedules/schedule-redeploy.json"
}

bert_flags() {
    echo -model "bert"
}

gpt_flags() {
    echo -model "gpt"
}

openwebtext_flags() {
    echo -dataset "openwebtext"
    echo -index-url "/data/megatron-lm/bert/openwebtext/npzs_seq1024/indices.txt"
}

comb_flags() {
    echo -hosts $(join $(hosts))

    echo -schedule $(join $(schedules))
    echo -model-sizes $(join $(model_sizes))
    echo -batch-sizes $(join $(batch_sizes))
    echo -micro-batch-sizes $(join $(micro_batch_sizes))
    # echo -redeploy

    echo -para-config "$(dirname $0)/para-config.json"

    # echo -dryrun
}

run_bert() {
    ./bin/multi-experiment $(base_flags) $(bert_flags) $(openwebtext_flags) $(comb_flags)
}

run_gpt() {
    ./bin/multi-experiment $(base_flags) $(gpt_flags) $(openwebtext_flags) $(comb_flags)
}

run_bert
run_gpt

# >multi-experiment.log 2>&1
