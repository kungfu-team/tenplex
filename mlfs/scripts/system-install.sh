#!/bin/sh

set -e

cd $(dirname $0)/..

export PATH=/usr/local/go/bin:$PATH

make

rm -fr build
./scripts/pack.sh

set +e
echo "stopping mlfsd"
sudo systemctl stop mlfs
echo "stopped mlfsd"
set -e

sudo dpkg -i ./build/*.deb
sudo systemctl daemon-reload

sudo systemctl start mlfs

echo "done $0"
