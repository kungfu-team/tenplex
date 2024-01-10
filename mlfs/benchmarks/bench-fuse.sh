#!/bin/sh
set -e

make

root=$HOME/mnt/efs
./bin/mlfs-test -mnt $root
