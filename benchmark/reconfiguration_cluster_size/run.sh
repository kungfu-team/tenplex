#!/bin/bash
set -e

cd $(dirname $0)

now_sec() {
    date +%s
}

_show_duration() {
    echo "$1s"
}

measure() {
    echo "BEGIN $@"
    local begin=$(now_sec)
    $@
    local end=$(now_sec)
    local duration=$((end - begin))
    echo "END $@, took $(_show_duration $duration)" | tee -a measure.log
}

export PATH=$HOME/go/bin:$PATH

x() {
    echo "BGN $@"
    measure $@
    echo "END $@"
    echo
    echo
}

with_log_file() {
    local filename=$1
    shift
    echo "logging to file $filename $@"
    $@ 2>&1 | tee $filename
    echo "logged to $filename $ $@"
}

# . ../common-cloud.sh
. ../common.sh
. ./config.sh

list_hosts() {
    # az vmss nic list -g kungfu --vmss-name $name --query '[].ipConfigurations[0].privateIPAddress' -o table | sed 1,2d
    echo "10.10.10.1"
    echo "10.10.10.4"
}

training_flags() {
    base_flags

    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "large"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "https://tenplex.blob.core.windows.net/tenplexcontainer/gpt_enwiki_indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
    # echo -disable-ib
}

wait_cluster() {
    for i in $(seq 20); do
        x ./list-ips.sh
        sleep 5
    done
}

run_group() {
    local cluster_size=$1
    local n=$2
    local para_config_tp=$3

    # x ./scale-cluster.sh $cluster_size
    # x wait_cluster
    # x ./upgrade.sh

    local schedule="schedule_${n}.json"
    local hosts="$(join $(list_hosts))"
    if [ -z "$hosts" ]; then
        echo "no hosts available"
        return
    fi
    echo "using hosts: $hosts"

    x with_log_file reconfig_${n}_dp.log tenplex-run $(training_flags) -schedule-file $schedule -para-config para-config-dp.json
    mv logs logs_${n}_dp

    x with_log_file reconfig_${n}_pp.log tenplex-run $(training_flags) -schedule-file $schedule -para-config para-config-pp.json
    mv logs logs_${n}_pp

    x with_log_file reconfig_${n}_tp.log tenplex-run $(training_flags) -schedule-file $schedule -para-config $para_config_tp
    mv logs logs_${n}_tp

    # x ./scale-cluster.sh 0
}

run_all() {
    # x ./recreate-vmss.sh
    x run_group 2 8 para-config-tp-4to8.json
    # x run_group 4 16 para-config-tp-8to16.json
    # x run_group 8 32 para-config-tp-16to32.json
}

x run_all

python plot.py
