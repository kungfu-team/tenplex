#!/bin/bash
set -e

export GO=/usr/local/go/bin/go

make

join_() {
    local IFS=","
    echo "$*"
}

w1=10.0.0.9
w2=10.0.0.8

ips() {
    echo $w1
    echo $w2
}

flags() {
    echo -gpu-per-host 4
    echo -hosts "$(join_ $(ips))"
}

# echo
./bin/tenplex-user $(flags)
