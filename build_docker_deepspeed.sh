#!/bin/bash
set -e

./Dockerfile-deepspeed

docker push kungfu.azurecr.io/deepspeed-tenplex:latest
