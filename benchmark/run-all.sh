#!/bin/sh
set -e

now_sec() {
    date +%s
}

measure() {
    echo "[->] $@ begins"
    local begin=$(now_sec)
    "$@"
    local end=$(now_sec)
    local duration=$((end - begin))
    echo "[==] $@ took ${duration}s" | tee -a time.log
}

cd $(dirname $0)

run_all() {
    measure ./run-megatron-lm-gpt-no-tenplex.sh
    measure ./run-megatron-lm-gpt-redeploy.sh
}

measure run_all
