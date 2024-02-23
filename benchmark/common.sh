#!/bin/sh

join_() {
    local IFS=$1
    shift
    echo "$*"
}

join() { join_ , $@; }

logfile="$(basename $0).log"

base_flags() {
    echo -image "kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest"
    echo -user $USER

    echo -mlfs-port 20010
    echo -tenplex-prefix "$HOME/.tenplex"

    # echo -logfile
}

tenplex_run_with() {
    local flags=$1
    tenplex-run $(flags) >$logfile 2>&1
}
