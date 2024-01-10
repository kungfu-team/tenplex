#!/bin/sh
set -e

mkdir -p $HOME/local
cd $HOME/local

wget https://dl.google.com/go/go1.18.linux-amd64.tar.gz
tar -xf go1.18.linux-amd64.tar.gz
