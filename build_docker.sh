#!/bin/bash
set -e

./Dockerfile

docker push kungfu.azurecr.io/mw-pytorch1-tenplex:latest
