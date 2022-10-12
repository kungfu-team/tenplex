#!/bin/bash
set -e

make

python save_ckpt.py \
    --mlfs-path /data/marcel/mlfs \
    --ckpt-path /data/marcel/training/0/ckpt/global_step50/mp_rank_00_model_states.pt
