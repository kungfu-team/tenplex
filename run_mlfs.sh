#!/bin/sh
set -e

flags() {
    echo -http-port 19999
    echo -ctrl-port 20010
    echo -mnt /mnt/mlfs
    echo -tmp /tmp/mlfs
    echo -su
    echo -log-req
}

sudo ./bin/mlfsd $(flags)

echo "$0 stopped"
