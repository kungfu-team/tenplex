#!/bin/bash
set -e

python -u mnist.py 2>&1 | tee mnist.log
python -u mnist.py --inconsistent-dataset 2>&1 | tee inconsistent.log

python plot.py
