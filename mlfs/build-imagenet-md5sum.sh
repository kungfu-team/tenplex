#!/bin/sh
set -e

make

cd $(dirname $0)

list_tf_records() {
    # SSD/HDD
    ls /data/imagenet/records/train* | sort

    # tmpfs
    # ls $HOME/mnt/tmp/train* | sort

    # NVMe
    #ls $HOME/data/train* | sort
}

./bin/mlfs-md5sum -m 64 -output imagenet.md5.txt $(list_tf_records)
