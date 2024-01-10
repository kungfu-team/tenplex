#!/bin/sh
set -e

make

TENPLEXPRE="$HOME/.tenplex"
root="$TENPLEXPRE/mlfs"

CACHE="$TENPLEXPRE/cache"
if [ ! -d $CACHE ]; then
    mkdir -p $CACHE
fi

flags() {
    echo -http-port 20005
    echo -ctrl-port 20010
    echo -mnt $root
    echo -tmp $CACHE
    echo -su
}

echo "starting mlfsd ..."
$TENPLEXPRE/bin/mlfsd $(flags)
