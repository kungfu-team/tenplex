#!/bin/bash
set -e

export GO=/usr/local/go/bin/go

echo "Building scheduler ..."
make all
mkdir -p $HOME/.tenplex/scheduler/bin
cp -v ./vendors/tenplex-run/mlfs/bin/mlfsd $HOME/.tenplex/scheduler/bin

echo "Running scheduler ..."
./bin/tenplex-scheduler
