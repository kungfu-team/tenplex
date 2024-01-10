#!/bin/bash
set -e

HOSTS="155.198.152.18 155.198.152.19 155.198.152.23"

for host in $HOSTS; do
	ssh $host mkdir -p ~/.tenplex/bin
	scp $HOME/Elasticity/Repo/state-migrator/go/bin/state-migrator $host:~/.tenplex/bin/state-migrator &
	scp -r $HOME/Elasticity/Repo/transformer-checkpoint $host:~/.tenplex &
	ssh $host sudo systemctl restart mlfs &
done

wait
