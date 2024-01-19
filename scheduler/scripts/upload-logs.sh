#!/bin/sh
set -e

export PATH=$PATH:$HOME/local/bin

cd $(dirname $0)/..

if [ ! -f run-id.txt ]; then
    date +%s >run-id.txt
fi

RUN_ID=$(cat run-id.txt)

echo "Using RUN_ID: $RUN_ID"

SA=tenplex
PREFIX="https://$SA.blob.core.windows.net/public/_debug/scheduler/$RUN_ID"

_list_logs() {
    find logs -type f
    ls *.log
}

list_logs() { _list_logs | sort; }

upload() {
    URL=$PREFIX/$1
    ucp $1 $URL
    echo "uploaded to $URL"
}

main() {
    for f in $(list_logs); do
        echo $f
        upload $f
    done
    ./scripts/gen-log-index.py $(list_logs) >index.html
    upload index.html
}

main
