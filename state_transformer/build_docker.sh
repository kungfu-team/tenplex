#!/bin/bash
set -e

make

./Dockerfile

docker push kungfu.azurecr.io/mw-megatron-lm-go:latest
