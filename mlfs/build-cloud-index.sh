#!/bin/sh
set -e

cd $(dirname $0)

host=minddata.blob.core.windows.net

list_tf_records() {
    for i in $(seq 1024); do
        echo https://$host/data/imagenet/records/train-$(printf "%05d" $((i - 1)))-of-01024
    done
}

./bin/mlfs-build-tf-index -m 8 -output imagenet.idx.txt $(list_tf_records)
