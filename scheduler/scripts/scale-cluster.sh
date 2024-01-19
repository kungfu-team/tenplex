#!/bin/sh
set -e

cd $(dirname $0)/..
. ./scripts/config.sh

n="$1"
az vmss scale -g $group -n $name --new-capacity $n -o table

echo "scaled to $n"
