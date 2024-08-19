#!/bin/bash
set -e

python -u mnist.py 2>&1 | tee mnist.log
python -u mnist.py --inconsistent-dataset 2>&1 | tee inconsistent_dataset.log
python -u mnist_batch_size.py 2>&1 | tee inconsistent_batch_size.log

python plot.py
