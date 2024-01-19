#!/bin/sh
set -e

make

./bin/mlfs mount -index-url http://155.198.152.18:20110/ -idx-name a -ctrl-port 9999
./bin/mlfs bench -mnt ./tmp
