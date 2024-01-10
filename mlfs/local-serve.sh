#!/bin/sh
set -e

make

# ./bin/mlfs-check-index $(cat tests/data/*.json | jq -r '."index-url"')

# ./bin/mlfs-debug -ds ./tests/data/squad1.json
# ./bin/mlfs-debug -ds ./tests/data/imagenet.json

./bin/mlfs-edit-index \
    -index-url $(cat tests/data/imagenet.json | jq -r '."index-url"') \
    -o a.index.txt \
    -localize

./bin/mlfs-check-index ./a.index.txt
./bin/mlfs serve -index-url ./a.index.txt -self 155.198.152.18
# ./bin/mlfs daemon -ctrl-port 9999 -http-port 9998 -mnt ./tmp
