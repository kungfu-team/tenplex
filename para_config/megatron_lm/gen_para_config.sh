#!/bin/bash
set -e

python gen_para_config.py \
    --model gpt \
    --size large \
    --precision fp16 \
    --pp 2 \
    --tp 2 \
    --dp 2
