#!/bin/bash
set -e

GLOBAL_STEP=200
DIR="pp2_mp2_dp1_400"

# move checkpoints into one directory
if [[ -d /data/${USER}/united ]]
then
    rm -r /data/${USER}/united
fi
mkdir /data/${USER}/united
cp -r /data/${USER}/${DIR}/*/ckpt/iter_0000${GLOBAL_STEP}/* /data/${USER}/united
