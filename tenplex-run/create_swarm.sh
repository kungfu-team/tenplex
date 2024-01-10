#!/bin/bash
set -e

HOSTS="10.0.0.6 10.0.0.7 10.0.0.8"

for host in $HOSTS; do
	ssh $host "docker swarm join --token SWMTKN-1-4zlj2ycbe4thnzr958zm161ma7y58bfj0e9bbdjkw2q0i39d9m-bbt2e2ttg6utcdvrtqi3s0r7k 10.0.0.5:2377" &
done

wait
