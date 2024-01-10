#!/bin/sh
set -e

export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
cd $(dirname $0)/..
export CUDA_VISIBLE_DEVICES=0

./benchmarks/tf_read.py --fake-data 1
# ./benchmarks/tf_read.py
