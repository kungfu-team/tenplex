#!/bin/sh
set -e

cd $(dirname $0)/..

GOBIN=$PWD/bin go install -v ./...

rm -rf build
mkdir -p build
cd build

branch=$(git rev-parse --abbrev-ref HEAD)
rev=$(git rev-list --count HEAD)
commit=$(git rev-parse --short HEAD)
export VERSION="0.0.1-git-${branch}-rev${rev}-${commit}"

cmake ..
make package

dpkg -c *.deb
