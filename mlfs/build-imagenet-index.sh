#!/bin/sh
set -e

list_tf_records() {
    ls /data/imagenet/records/train* | sort
}

mlfs-build-tf-index -m 16 -output imagenet.idx.txt $(list_tf_records)
