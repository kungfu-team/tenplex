#!/bin/sh
set -e

make

# ./mlfs.sh

python3.8 test-tensor-file.py
