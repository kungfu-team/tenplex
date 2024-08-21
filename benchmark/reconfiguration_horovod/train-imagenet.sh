#!/bin/sh
set -e

run() {
    local np=$1
    shift
    horovodrun -np $np $@
}

train_flags_disk() {
    echo --data-dir /data/imagenet/records
}

train_flags_tenplex() {
    echo --mlfs-dir /mnt/mlfs
    echo --job fig-13
}

with_log_file() {
    local filename=$1
    shift
    $@ | tee $filename
    echo "logged to $filename $ $@"
}

with_log_file 1.log run 2 python3 ./imagenet_resnet.py $(train_flags_disk)
with_log_file 3.log run 2 python3 ./imagenet_resnet_horovod_elastic.py $(train_flags_disk)
with_log_file 2.log run 2 python3 ./imagenet_resnet.py $(train_flags_tenplex)
