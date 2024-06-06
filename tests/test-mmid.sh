#!/bin/sh
set -e

cd $(dirname $0)

common_flags() {
    echo --data-dir "/mnt/k1d2/megatron-lm"
}

bert_flags() {
    common_flags
    echo --dataset "enwiki"
    echo --model "bert"

    # <path>.idx, <path>.bin
    # echo --data-path /data/dataset/bert_text_sentence
    echo --data-path /data/megatron-lm/bert/enwiki/bert_text_sentence

    # /data/megatron-lm/bert/enwiki/bert_text_sentence.idx

}

gpt_flags() {
    common_flags
    echo --model "gpt"
    echo --data-path /data/dataset/gpt-2/my-gpt2_text_document
}

flags() {
    bert_flags
    # gpt_flags
}

python3 read_mmid.py $(flags)
