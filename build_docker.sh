#!/bin/bash
set -e

docker pull kungfu.azurecr.io/mw-megatron-lm-23.06:latest

./Dockerfile

docker push kungfu.azurecr.io/mw-megatron-lm-23.06-tenplex:latest
