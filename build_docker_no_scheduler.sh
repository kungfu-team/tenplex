#!/bin/bash
set -e

./Dockerfile-no-scheduler

docker push kungfu.azurecr.io/mw-megatron-lm-no-scheduler-tenplex:latest
