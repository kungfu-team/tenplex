#!/bin/sh
set -e

make
# ./bin/mlfs mount

peer1="127.0.0.1:8080"
peer2="127.0.0.1:8081"

peer_flags() {
    echo -r 1
    echo -peers "$peer1,$peer2"
    echo -log-req
}

localhost="127.0.0.1"

./bin/mlfs daemon $(peer_flags) -host $localhost -ctrl-port 8080 -http-port 10000 &
pid1=$!
echo $p1

./bin/mlfs daemon $(peer_flags) -host $localhost -ctrl-port 8081 -http-port 10001 &
pid2=$!
echo $p2

wait
