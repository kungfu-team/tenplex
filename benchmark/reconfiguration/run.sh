#!/bin/bash
set -e

sudo rm -rf /mnt/k1d2/ckpt/*
python training.py --scale-up

# python plot.py
