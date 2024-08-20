#!/bin/sh
set -e

run() {
    local np=$1
    shift
    horovodrun -np $np $@
}

train_flags_tenplex() {
    echo --mlfs-dir /mnt/mlfs
    echo --job fig-13
    # echo --num-iters 100
}

train_flags_disk() {
    # echo --data-dir /data/imagenet/records
    true
}

# run 2 python3 ./imagenet_resnet.py $(train_flags_tenplex)

# run 2 python3 ./imagenet_resnet.py $(train_flags_disk)

run 2 python3 ./imagenet_resnet_horovod_elastic.py $(train_flags_disk)
