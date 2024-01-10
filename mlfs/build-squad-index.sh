#!/bin/sh
set -e

cd $(dirname $0)

list_squad_records() {
    if [ $(uname) = "Darwin" ]; then
        echo $HOME/squad1_train.tf_record
    else
        echo /data/squad1/train.tf_record
    fi
}

./bin/mlfs-build-tf-index $(list_squad_records)
