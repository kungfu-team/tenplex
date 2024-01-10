#!/bin/bash
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

mlfs_git_commit="73406233c1c8e96db5af9b685bde8a076b6f8822"

run() {
    local failure=$1
    local log_folder=recover-$failure
    mkdir -p $log_folder

    ./bin/tenplex-run \
        -mlfs-git-commit "$mlfs_git_commit" \
        -image "kungfu.azurecr.io/mw-megatron-lm-update:latest" \
        -framework "megatron-lm" \
        -model "gpt" \
        -model-size "2.7B" \
        -dataset "enwiki" \
        -batch-size 128 \
        -micro-batch-size 8 \
        -precision "fp16" \
        -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt" \
        -hosts "$(join $(hosts))" \
        -tenplex-prefix "$HOME/.tenplex" \
        -schedule-file "$(pwd)/data/schedule-failure.json" \
        -mlfs-port 20010 \
        -gpu-per-host 4 \
        -gpu-per-container 4 \
        -user $USER \
        -failure $failure \
        -seq-length 1024 >$log_folder/tenplex-run.log 2>&1
    mv logs $log_folder
}

# run 0

run 4
run 8
run 12

# for i in $(seq 3); do
#     run $i
# done
