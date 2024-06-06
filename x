#!/bin/sh

set -e

./lint.sh
./tests/test-mmid.sh

# make install
# cd benchmark/dynamic_resources
# ./run.sh
git add -A

echo "done $0"
