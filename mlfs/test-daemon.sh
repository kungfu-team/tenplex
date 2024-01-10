#!/bin/sh
set -e

make

./bin/mlfs daemon -ctrl-port 9999 -http-port 9998 -mnt ./tmp
