#!/bin/sh
set -e

cd $(dirname $0)/..

docker build --rm -t kungfu.azurecr.io/mw-pytorch2:latest -f ./docker/Dockerfile.base-1 docker

# TODO:
# docker build --rm -t kungfu.azurecr.io/mw-megatron-lm-23.06:latest -f ./docker/Dockerfile.base-2 docker
