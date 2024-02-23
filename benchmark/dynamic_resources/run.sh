#!/bin/bash

scheduler_flags() {
    echo -detect-self-ip ib0
    echo -reinstall
    echo -u marcel
}

tenplex-scheduler $(scheduler_flags) &
pid=($!)

join_() {
    local IFS=$1
    shift
    echo "$*"
}

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    echo "10.10.10.3"
    echo "10.10.10.4"
}

hosts=$(join_ , $(list_hosts))

user_flags() {
    echo -hosts $hosts
    echo -gpu-per-host 4
    echo -image kungfu.azurecr.io/mw-megatron-lm-update
    echo -plan ./single-job-time.json
    echo -timed-job
}

tenplex-user $(user_flags)

# Kill scheduler after user finishes
pkill -P $$
