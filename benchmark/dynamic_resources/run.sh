#!/bin/bash
set -e

scheduler_common_flags() {
    echo -detect-self-ip ib0
    echo -u $USER
    echo -reinstall
}

scheduler_tenplex_flags() {
    # echo -device-allocation ./tenplex-allocation.json
    echo -device-allocation ./tenplex-allocation-test.json
}

scheduler_pytorch_flags() {
    # echo -device-allocation ./pytorch-allocation.json
    echo -device-allocation ./pytorch-allocation-test.json
}


join_() {
    local IFS=$1
    shift
    echo "$*"
}

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
    # echo "10.10.10.3"
    # echo "10.10.10.4"
}

hosts=$(join_ , $(list_hosts))

user_common_flags() {
    echo -hosts $hosts
    echo -gpu-per-host 4
    echo -image kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest
    echo -timed-job
}

user_tenplex_flags() {
    # echo -plan ./tenplex-schedule.json
    echo -plan ./tenplex-schedule-test.json
}

user_pytorch_flags() {
    # echo -plan ./pytorch-schedule.json
    echo -plan ./pytorch-schedule-test.json
}

tenplex-scheduler $(scheduler_common_flags) $(scheduler_tenplex_flags) &
tenplex-user $(user_common_flags) $(user_tenplex_flags)
pkill -P $$

# tenplex-scheduler $(scheduler_common_flags) $(scheduler_pytorch_flags) &
# tenplex-user $(user_common_flags) $(user_pytorch_flags)
# pkill -P $$
