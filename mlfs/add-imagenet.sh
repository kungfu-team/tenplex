#!/bin/sh
set -e

flags() {
    echo -idx-name imagenet
    echo -index-url https://tenplex.blob.core.windows.net/data/imagenet.idx.txt
    echo -ctrl-port 20000

    echo -progress 0
    echo -global-batch-size 23
    echo -dp-size 4

    # echo -fetch
    # echo -m 64
}

mlfs mount $(flags)
