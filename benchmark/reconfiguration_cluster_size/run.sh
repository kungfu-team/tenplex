#!/bin/bash
set -e

. ../common.sh
. ./config.sh

list_hosts() {
	az vmss nic list -g kungfu --vmss-name $name --query '[].ipConfigurations[0].privateIpAddress' -o table | sed 1,2d
}

training_flags() {
    base_flags

    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "xl"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "https://tenplex.blob.core.windows.net/tenplexcontainer/gpt_enwiki_indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
}

. ./recreate-vmss.sh

. ./scale-cluster.sh 2

tenplex-run $(training_flags) -schedule-file schedule-8.json -para-config para-config-dp.json 2>&1 | tee reconfig_8_dp.log
mv logs logs_8_dp
tenplex-run $(training_flags) -schedule-file schedule-8.json -para-config para-config-pp.json 2>&1 | tee reconfig_8_pp.log
mv logs logs_8_pp
tenplex-run $(training_flags) -schedule-file schedule-8.json -para-config para-config-tp.json 2>&1 | tee reconfig_8_tp.log
mv logs logs_8_tp

. ./scale-cluster.sh 4

tenplex-run $(training_flags) -schedule-file schedule-16.json -para-config para-config-dp.json 2>&1 | tee reconfig_16_dp.log
mv logs logs_16_dp
tenplex-run $(training_flags) -schedule-file schedule-16.json -para-config para-config-pp.json 2>&1 | tee reconfig_16_pp.log
mv logs logs_16_pp
tenplex-run $(training_flags) -schedule-file schedule-16.json -para-config para-config-tp.json 2>&1 | tee reconfig_16_tp.log
mv logs logs_16_tp

. ./scale-cluster.sh 8

tenplex-run $(training_flags) -schedule-file schedule-16.json -para-config para-config-dp.json 2>&1 | tee reconfig_32_dp.log
mv logs logs_32_dp
tenplex-run $(training_flags) -schedule-file schedule-16.json -para-config para-config-pp.json 2>&1 | tee reconfig_32_pp.log
mv logs logs_32_pp
tenplex-run $(training_flags) -schedule-file schedule-16.json -para-config para-config-tp.json 2>&1 | tee reconfig_32_tp.log
mv logs logs_32_tp

. ./scale-cluster.sh 0

python plot.py
