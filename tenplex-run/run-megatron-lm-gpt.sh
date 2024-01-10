#!/bin/bash
set -e

make

    # -image "kungfu.azurecr.io/mw-megatron-lm-update:latest" \
    # -image "kungfu.azurecr.io/mw-megatron-lm-no-scheduler-torch-save:latest" \
    # -index-url "https://tenplex.blob.core.windows.net/tenplexcontainer/gpt_enwiki_indices.txt" \
    # -hosts "10.10.10.1,10.10.10.2,10.10.10.3,10.10.10.4" \
    # -hosts "komodo01.doc.res.ic.ac.uk,komodo02.doc.res.ic.ac.uk,komodo03.doc.res.ic.ac.uk,komodo04.doc.res.ic.ac.uk" \
    # -hosts "155.198.152.18,155.198.152.17,155.198.152.19,155.198.152.23" \
./bin/tenplex-run \
    -image "kungfu.azurecr.io/mw-megatron-lm-update:latest" \
    -framework "megatron-lm" \
    -model "gpt" \
    -model-size "xl" \
    -dataset "enwiki" \
    -batch-size 128 \
    -micro-batch-size 8 \
    -precision "fp16" \
    -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt" \
    -hosts "10.10.10.1" \
    -tenplex-prefix "$HOME/.tenplex" \
    -schedule-file "$(pwd)/schedule.json" \
    -mlfs-port 20010 \
    -gpu-per-host 4 \
    -gpu-per-container 2 \
    -user $USER \
    -seq-length 1024 > tenplex-run.log 2>&1
