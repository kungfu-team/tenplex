#!/bin/sh
set -e

. ./config.sh

n="$1"
az vmss scale -g $group -n $name --new-capacity $n -o table

echo "scaled to $n"
