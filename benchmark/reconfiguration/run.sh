#!/bin/bash
set -e

sudo rm -rf /mnt/k1d2/ckpt/*
python -u training.py 2>&1 | tee run_scale_down.log

# sudo rm -rf /mnt/k1d2/ckpt/*
# python -u training.py --scale-up 2>&1 | tee run_scale_up.log

# python plot.py
