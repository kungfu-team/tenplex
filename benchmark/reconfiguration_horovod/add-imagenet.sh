#!/bin/sh
set -e

flags() {
    echo -idx-name imagenet
    echo -index-url /data/imagenet/imagenet.idx.txt

    echo -ctrl-port 20010

    echo -progress 0
    echo -global-batch-size 64
    echo -dp-size 1

    echo -job fig-13
}

sudo systemctl stop mlfs
sudo systemctl start mlfs

mlfs info
mlfs mount $(flags)
