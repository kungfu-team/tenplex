#!/bin/sh
set -e

cd $(dirname $0)/..
. ./scripts/config.sh

az vmss nic list -g kungfu --vmss-name $name --query '[].ipConfigurations[0].privateIpAddress' -o table -o table | sed 1,2d
