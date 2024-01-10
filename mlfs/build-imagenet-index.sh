#!/bin/sh
set -e

make

cd $(dirname $0)

list_tf_records() {
    ls /data/imagenet/records/train* | sort
}

./bin/mlfs-build-tf-index -m 16 -output imagenet.idx.txt $(list_tf_records)
