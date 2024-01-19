#!/bin/bash

HOSTS="155.198.152.18 155.198.152.19 155.198.152.23"

for host in $HOSTS; do
	ssh $host "docker ps -f \"name=trainer\" -q | xargs docker stop" &
	ssh $host "sudo rm -r ~/.tenplex/training/*" &
done

wait
