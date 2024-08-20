#!/usr/bin/env -S sh -c 'docker build --rm --no-cache -t $(cat $(dirname $0)/tag.txt) -f $0 $(dirname $0)'

FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

RUN python3 -m pip install tenplex -i https://pkgs.dev.azure.com/gli7/releases/_packaging/nightly/pypi/simple -U

WORKDIR /work
ADD imagenet.py .
ADD imagenet_resnet.py .
ADD imagenet_resnet_horovod_elastic.py .
ADD train-imagenet.sh .

ENV OMPI_ALLOW_RUN_AS_ROOT=1
