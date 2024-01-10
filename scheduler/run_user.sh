#!/bin/bash
set -e

cd $(dirname $0)
make

join_() {
    local IFS=$1
    shift
    echo "$*"
}

echo "Listing IPs"
# host=$(join_ , $(./scripts/list-ips.sh))
host=$(join_ , $(./scripts/list-ips-komodo.sh))
echo "using host=$host"

for h in $(echo $host | tr ',' '\n'); do
    gpu_per_host=$(ssh $h nvidia-smi -L | wc -l)
    echo "$gpu_per_host GPUs on $h"
done

    # echo -failure
flags() {
    echo -hosts $host
    echo -gpu-per-host $gpu_per_host # TODO: auto detect
    echo -image kungfu.azurecr.io/mw-megatron-lm-update
    echo -plan ./data/single-job-time.json
    echo -timed-job
}

PREFIX=$HOME/.tenplex/scheduler
$PREFIX/bin/tenplex-user $(flags)

echo "$0 done"
