#!/bin/sh

set -e

make

join_() {
    local IFS=$1
    shift
    echo "$*"
}

join() { join_ , $@; }

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

batch_sizes() {
    echo 128
    # echo 256
}

micro_batch_sizes() {
    echo 8
}

schedules() {
    # ls data/schedule-*.json | sort
    echo "data/schedule-redeploy.json"
}

flags() {
    echo -hosts $(join $(hosts))

    echo -schedule $(join $(schedules))
    echo -model-sizes $(join $(model_sizes))
    echo -batch-sizes $(join $(batch_sizes))
    echo -micro-batch-sizes $(join $(micro_batch_sizes))
    echo -redeploy

    echo -image kungfu.azurecr.io/mw-megatron-lm-update:latest

    # echo -dryrun
}

./bin/multi-experiment $(flags) > multi-experiment.log 2>&1
