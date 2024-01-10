#!/bin/sh
set -e

./bin/mlfs-fetch -ctrl-port 20000 -file 'https://minddata.blob.core.windows.net/data/squad1/train.tf_record' -md5 67eb6da21920dda01ec75cd6e1a5b8d7
