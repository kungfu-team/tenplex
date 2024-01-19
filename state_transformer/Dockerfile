#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm-go:latest -f $0 .'
FROM kungfu.azurecr.io/mw-megatron-lm-data-commit:latest

ADD go /usr/local
ENV PATH=$PATH:/usr/local/go/bin
RUN whereis go
RUN go version
ENV GO=/usr/local/bin/go

# State Transformer
ADD . /workspace/state_transformer
WORKDIR /workspace/state_transformer
RUN make
WORKDIR /workspace/Megatron-LM
