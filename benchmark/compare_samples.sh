#!/bin/bash
set -e

rm -rf logs logs_tde logs_tenplex training

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
    echo -schedule-file "./schedule.json"
    echo -para-config "./para-config.json"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
    echo -no-shuffle
}

sudo systemctl restart mlfs
sudo rm -rf $HOME/.tenplex/training/*
mkdir -p samples_tenplex
tenplex-run $(flags)
cp $HOME/.tenplex/training/*/0/ckpt/samples_*.txt samples_tenplex
mv logs logs_tenplex

sudo rm -f /mnt/k1d2/megatron-lm/gpt-2/enwiki/*.npy
sudo rm -f /mnt/k1d2/megatron-lm/gpt-2/*.npy
sudo rm -rf /mnt/k1d2/ckpt/*
mkdir -p samples_tde
tenplex-run $(flags) -no-tenplex
cp /mnt/k1d2/ckpt/samples_*.txt samples_tde
mv logs logs_tde

head -n 4 samples_tenplex/*
head -n 4 samples_tde/*
