#!/bin/bash
set -e

GLOBAL_STEP=100
DIR="pp4_mp1_dp1_200"

# move checkpoints into one directory
if [[ -d /data/${USER}/united ]]
then
    rm -r /data/${USER}/united
fi
mkdir /data/${USER}/united
cp /data/${USER}/${DIR}/*/ckpt/global_step${GLOBAL_STEP}/* /data/${USER}/united
