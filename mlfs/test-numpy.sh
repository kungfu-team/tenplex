#!/bin/sh
set -e

make

# DATA_DIR=/data/megatron-lm/bert/npz_concat
DATA_DIR=/data/megatron-lm/bert/test

flags() {
    # echo -index-file $DATA_DIR/indices.txt
    echo -index-file $DATA_DIR/old-format.txt
    # echo -data-file $DATA_DIR/samples.npzs

    echo -dp-size 4
    echo -global-batch-size 32
}

./bin/mlfs-gen-numpy $(flags)
