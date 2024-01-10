#!/bin/sh
set -e

cd $(dirname $0)/..
. ./scripts/config.sh

# The Image type for a Virtual Machine Scale Set may not be changed.
# image=tenplex-base-image
image="tenplex-mw"
image=$(az image show -n $image -g kungfu | jq -r .id)
echo "Using image $image"

storage=Premium_LRS # SSD

flags() {
    echo --admin-username kungfu
    echo --vnet-name tenplex-relayVNET
    echo --subnet tenplex-relaySubnet
    echo --disable-overprovision
    echo --image $image
    echo --instance-count 0
    echo --vm-sku $size
    echo --location westeurope
    echo --storage-sku $storage
    # echo --lb '""'
}

recreate() {
    az vmss delete -g $group -n $name
    echo "deleted $name"

    az vmss create -g $group -n $name $(flags) --lb ""
    echo "created $name"
}

recreate
