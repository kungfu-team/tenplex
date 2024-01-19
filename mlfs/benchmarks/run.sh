#!/bin/sh

set -e

export PYTHON=$(which python3.6)

cd $(dirname $0)

kungfu_run_flags() {
    echo -q
    echo -logdir logs/$JOB_ID
    echo -np 4
}

kungfu_run() {
    echo "JOB_ID: $JOB_ID"
    kungfu-run $(kungfu_run_flags) $@
}

flags_mount() {
    echo --index-file $HOME/tf-index-16.idx.txt
    echo --seed 1
    echo --global-batch-size 128
    echo --tfrecord-fs $PWD/../../bin/tfrecord-fs
}

flags() {
    flags_mount
    echo --run-train-op
}

flags_baseline() {
    flags
    echo --prefix $HOME/mnt/all
}

flags_fake_data() {
    flags
    echo --prefix $HOME/mnt/all
    echo --fake-data
}

flags_vfs() {
    flags
}

main() {
    export JOB_ID=vfs
    kungfu_run $PYTHON train_resnet50.py $(flags_vfs)

    # export JOB_ID=base
    # kungfu_run $PYTHON train_resnet50.py $(flags_baseline)

    # export JOB_ID=fakedata
    # kungfu_run $PYTHON train_resnet50.py $(flags_fake_data)
}

rm -fr *.log
main
