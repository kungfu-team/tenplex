#!/bin/bash
set -e

GLOBAL_STEP=1

# copy checkpoints from other machines
scp -r komodo3:/data/${USER}/training/* /data/${USER}/training

# move checkpoints into one directory
if [[ -d /data/${USER}/united ]]
then
    rm -r /data/${USER}/united
fi
mkdir /data/${USER}/united
cp /data/${USER}/training/*/ckpt/global_step${GLOBAL_STEP}/* /data/${USER}/united

# create transformed directory
if [[ -d /data/${USER}/transformed ]]
then
    rm -r /data/${USER}/transformed
fi
mkdir /data/${USER}/transformed
