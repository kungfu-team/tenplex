#!/bin/bash
set -e

make binaries

sudo rm -rf ~/.tenplex/training/*
sudo rm -f /mnt/k1d2/megatron-lm/gpt-2/enwiki/*.npy
sudo rm -f /mnt/k1d2/megatron-lm/gpt-2/*.npy

flags() {
    echo -image "kungfu.azurecr.io/mw-megatron-lm-23.06-debug:latest"
    echo -user $USER
    echo -mlfs-port 20010
    echo -tenplex-prefix "$HOME/.tenplex"
    echo -framework "megatron-lm"
    echo -model "gpt"
    echo -model-size "large"
    echo -dataset "enwiki"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt"
    echo -hosts "10.10.10.2"
    echo -schedule-file "./benchmark/schedule.json"
    echo -para-config "./benchmark/para-config.json"
    echo -gpu-per-host 1
    echo -gpu-per-container 1
    echo -seq-length 1024
}

./bin/tenplex-debug $(flags)
cp ~/.tenplex/training/tenplexdeb/0/ckpt/samples_build.txt ./samples_tenplex.txt

./bin/tenplex-debug $(flags) -no-tenplex
cp ~/.tenplex/training/tenplexdeb/0/ckpt/samples_build.txt ./samples_tde.txt

head samples_*.txt
