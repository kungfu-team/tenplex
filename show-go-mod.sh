#!/bin/sh
set -e

cat go.mod | head -n 1 | awk '{print $2}'
