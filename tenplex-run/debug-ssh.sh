#!/bin/sh
set -e

make
./bin/elastique-test-ssh
