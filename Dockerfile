#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm-tenplex:latest -f $0 .'

FROM kungfu.azurecr.io/mw-megatron-lm-commit:latest

# Tenplex
ADD . /workspace/tenplex
RUN cd /workspace/tenplex && \
    pip install .

# Megatron-LM
WORKDIR /workspace/Megatron-LM
