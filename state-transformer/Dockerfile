#!/usr/bin/env -S sh -c ' docker build --rm -t kungfu.azurecr.io/deepspeed-060:latest .'

FROM kungfu.azurecr.io/deepspeed-mdp:latest

# Training state
ADD . /workspace/training-state
