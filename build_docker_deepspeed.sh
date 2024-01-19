#!/bin/bash
set -e

./Dockerfile-deepspeed

docker push kungfu.azurecr.io/mw-deepspeed-tenplex:latest
