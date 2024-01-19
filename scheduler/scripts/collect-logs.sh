#!/bin/sh
set -e

cd $(dirname $0)/..

collect_logs() {
    mkdir -p logs
    for ip in $(./scripts/list-ips.sh); do
        echo $ip
        scp -r $ip:.tenplex/training logs
    done
}

main() {
    for h in $(./scripts/list-ips.sh); do
        echo $h
        ssh $h find /mnt/mlfs | tee logs/$h.mlfs.log
    done

    collect_logs
}

main
