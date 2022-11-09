#!/bin/bash
set -e

./Dockerfile

docker push kungfu.azurecr.io/mw-megatron-lm-tenplex:latest
