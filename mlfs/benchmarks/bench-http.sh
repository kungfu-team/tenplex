#!/bin/sh
set -e

make

./bin/mlfs-test -port 19999
