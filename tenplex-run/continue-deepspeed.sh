#!/bin/bash
set -e

make

HOST_PATH=/data/${USER}/training

./bin/elastique-run \
    -final-clean \
    -host-path ${HOST_PATH} \
    -image "kungfu.azurecr.io/deepspeed-mdp:latest" \
    -framework "deepspeed" \
    -model "gpt-2" \
    -dataset "enwiki" \
    -batch-size 32 \
    -precision "fp32" \
    -seq-length 1024 > elastique-run.log 2>&1
