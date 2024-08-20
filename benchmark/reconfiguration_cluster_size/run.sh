#!/bin/bash

. ./recreate-vmss.sh

. ./scale-cluster.sh 2

. ./scale-cluster.sh 0
