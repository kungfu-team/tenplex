#!/bin/bash
set -e

GLOBAL_STEP=10
OUTPUT_DIR="/data/marcel/debug/global_step${GLOBAL_STEP}"
SHAPES_DIR="./scripts/shapes"
SOURCE_MP_SIZE=4
TARGET_MP_SIZE=1
RANK=0 # starting from 0

mkdir -p ${OUTPUT_DIR}

python ./training_state/repartition.py \
    --output-dir ${OUTPUT_DIR} \
    --shapes-dir ${SHAPES_DIR} \
    --source-mp-size ${SOURCE_MP_SIZE} \
    --target-mp-size ${TARGET_MP_SIZE} \
    --rank ${RANK} \
    --global-step ${GLOBAL_STEP} \
    --hostfile "./debug_hostfile.txt" \
    --port 21234
