#!/bin/sh
set -e

make

make sys-install
make reload

./bin/mlfs info

# ./bin/mlfs-cli -sas "minddata:$(cat $HOME/.az/minddata.sas)"

./bin/mlfs mount -global-batch-size 23 -dp-size 4 \
    -idx-name squad1 \
    -index-url https://minddata.blob.core.windows.net/data/squad1/squad1.idx.txt

./bin/mlfs fetch -file 'https://minddata.blob.core.windows.net/data/squad1/train.tf_record' -md5 67eb6da21920dda01ec75cd6e1a5b8d7

sleep 1 # 2023/01/16 10:00:56 open /mnt/mlfs/job/0/head.txt: transport endpoint is not connected
./bin/mlfs bench -mnt /mnt/mlfs

tree /mnt/mlfs/
