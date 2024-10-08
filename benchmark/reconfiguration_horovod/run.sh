#!/bin/sh
set -e

cd $(dirname $0)

with_log_file() {
    local filename=$1
    shift
    $@ | tee $filename
    echo "logged to $filename $ $@"
}

./add-imagenet.sh

with_log_file 1.log ./with-docker horovodrun -np 2 python3 ./imagenet_resnet.py --data-dir /data/imagenet/records
with_log_file 2.log ./with-docker horovodrun -np 2 python3 ./imagenet_resnet_horovod_elastic.py --data-dir /data/imagenet/records
with_log_file 3.log ./with-docker horovodrun -np 2 python3 ./imagenet_resnet.py --mlfs-dir /mnt/mlfs --job fig-13

python3 plot.py
