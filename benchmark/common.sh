#!/bin/sh

join_() {
    local IFS=$1
    shift
    echo "$*"
}

join() { join_ , $@; }

logfile="$(basename $0).log"

base_flags() {
    echo -image "kungfu.azurecr.io/mw-megatron-lm-23.06-update:v0.0.3"

    echo -user $USER

    echo -mlfs-port 20010
    echo -tenplex-prefix "$HOME/.tenplex"

    # echo -logfile
}

tenplex_run_with() {
    tenplex-run $($1) 2>&1 | tee $logfile
}
