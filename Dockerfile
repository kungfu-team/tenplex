#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-pytorch1-tenplex:latest -f $0 .'

FROM kungfu.azurecr.io/mw-pytorch1:latest

# Tenplex
ADD . /workspace/tenplex
RUN cd /workspace/tenplex && \
    pip install .
