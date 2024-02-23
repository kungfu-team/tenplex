#!/bin/sh

join_() {
    local IFS=$1
    shift
    echo "$*"
}

join() { join_ , $@; }

base_flags() {
    echo -image "kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest"
    echo -user $USER

    echo -mlfs-port 20010
}
