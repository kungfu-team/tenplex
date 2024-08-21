#!/bin/bash
set -e

cd $(dirname $0)

. ./config.sh
echo $name

list_hosts() {
    az vmss nic list -g kungfu --vmss-name $name --query '[].ipConfigurations[0].privateIPAddress' -o table | sed 1,2d
}

list_hosts | tee hosts.txt

