#!/usr/bin/env -S sh -c 'docker build --rm--build-arg SSH_KEY="${SSH_KEY}"  -t $(cat $(dirname $0)/tag.txt) -f $0 $(dirname $0)/../../..'

FROM ubuntu:jammy AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y golang-go make git cmake

RUN mkdir $HOME/.ssh
RUN echo "StrictHostKeyChecking no" >$HOME/.ssh/config
ARG SSH_KEY
RUN echo "${SSH_KEY}" >$HOME/.ssh/id_rsa
RUN chmod 0600 $HOME/.ssh/id_rsa
RUN ssh-keygen -y -f $HOME/.ssh/id_rsa >$HOME/.ssh/id_rsa.pub

RUN git config --global url."git@github.com:".insteadOf "https://github.com/"
RUN go env -w GOPRIVATE=*

WORKDIR /src
ADD . .
RUN GOBIN=$PWD/bin go install -v ./...
RUN ./scripts/pack.sh
RUN cp ./build/*.deb mlfs.deb

FROM ubuntu:jammy

RUN apt update
RUN apt install -y systemd init fuse

COPY --from=builder /src/mlfs.deb /
RUN dpkg -i /mlfs.deb && rm /mlfs.deb
