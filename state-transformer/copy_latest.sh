#!/bin/bash

set -e

SOURCE_FILE=/data/${USER}/ckpt/0/latest

cp ${SOURCE_FILE} /data/${USER}/ckpt/1
scp ${SOURCE_FILE} komodo02:/data/${USER}/ckpt/2
scp ${SOURCE_FILE} komodo02:/data/${USER}/ckpt/3

###
SOURCE_FILE=/data/${USER}/ckpt/0/latest_checkpointed_iteration.txt

cp ${SOURCE_FILE} /data/${USER}/ckpt/1
scp ${SOURCE_FILE} komodo02:/data/${USER}/ckpt/2
scp ${SOURCE_FILE} komodo02:/data/${USER}/ckpt/3
