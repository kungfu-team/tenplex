#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

GLOBAL_STEP=2000
SOURCE_MP_SIZE=2
TARGET_MP_SIZE=4
# OUTPUT_DIR="/data/marcel/repartition/${SOURCE_MP_SIZE}to${TARGET_MP_SIZE}/global_step${GLOBAL_STEP}"
OUTPUT_DIR="/data/marcel/repartition/${TARGET_MP_SIZE}to${SOURCE_MP_SIZE}to${TARGET_MP_SIZE}/global_step${GLOBAL_STEP}"
SHAPES_DIR="./scripts/shapes"

mkdir -p ${OUTPUT_DIR}

max_rank=$((${TARGET_MP_SIZE}-1))
for rank in $(seq 0 ${max_rank})
do
    echo $rank
    python ./scripts/repartition.py \
        --output-dir ${OUTPUT_DIR} \
        --shapes-dir ${SHAPES_DIR} \
        --source-mp-size ${SOURCE_MP_SIZE} \
        --target-mp-size ${TARGET_MP_SIZE} \
        --global-step ${GLOBAL_STEP} \
        --rank ${rank}
done
