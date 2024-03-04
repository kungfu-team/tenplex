#!/bin/sh
set -e

make

flags() {
    echo -idx-name imagenet
    echo -idx-file https://minddata.blob.core.windows.net/data/imagenet.idx.txt
    echo -ctrl-port 20000

    echo -progress 0
    echo -global-batch-size 23
    echo -cluster-size 4

    echo -fetch

    echo -m 64
}

./bin/mlfs mount $(flags)
