#!/bin/bash
set -e

. $(dirname $0)/../common.sh

list_hosts() {
    echo "10.10.10.1"
    echo "10.10.10.2"
}

flags() {
    base_flags
    echo -framework "megatron-lm"
    echo -model "bert"
    echo -model-size "large"
    echo -dataset "openwebtext"
    echo -batch-size 128
    echo -micro-batch-size 8
    echo -precision "fp16"
    echo -index-url "/data/megatron-lm/bert/openwebtext/npzs_seq1024/indices.txt"
    echo -hosts "$(join $(list_hosts))"
    echo -gpu-per-host 4
    echo -gpu-per-container 4
    echo -seq-length 1024
}

tenplex-run $(flags) -schedule-file "schedule-static.json" -para-config "para-config-dp.json" -jobid "static" 2>&1 | tee static.log
mv logs logs-static

tenplex-run $(flags) -schedule-file "schedule-up.json" -para-config "para-config-dp.json" -jobid "dp-up" 2>&1 | tee dp-up.log
mv logs logs-dp-up

tenplex-run $(flags) -schedule-file "schedule-down.json" -para-config "para-config-dp.json" -jobid "dp-down" 2>&1 | tee dp-down.log
mv logs logs-dp-down

tenplex-run $(flags) -schedule-file "schedule-up.json" -para-config "para-config-tp.json" -jobid "tp-up" 2>&1 | tee tp-up.log
mv logs logs-tp-up

tenplex-run $(flags) -schedule-file "schedule-down.json" -para-config "para-config-tp.json" -jobid "tp-down" 2>&1 | tee tp-down.log
mv logs logs-tp-down

tenplex-run $(flags) -schedule-file "schedule-up.json" -para-config "para-config-pp.json" -jobid "pp-up" 2>&1 | tee pp-up.log
mv logs logs-pp-up

tenplex-run $(flags) -schedule-file "schedule-down.json" -para-config "para-config-pp.json" -jobid "pp-down" 2>&1 | tee pp-down.log
mv logs logs-pp-down

python plot.py
