#!/bin/sh
set -e

make

flags() {
    echo -idx-name enwiki
    echo -idx-file /data/megatron-lm/bert/enwiki/npzs_seq512/indices.txt
    echo -ctrl-port 20010

    echo -progress 0
    echo -global-batch-size 32
    echo -cluster-size 4

    # echo -fetch

    echo -m 64
}

./bin/mlfs mount $(flags)
