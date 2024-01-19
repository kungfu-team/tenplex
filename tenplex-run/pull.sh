#!/bin/bash
set -e

HOSTS="10.0.0.5 10.0.0.6 10.0.0.8 10.0.0.9"
JOBID="cd1e6f634c"

mkdir -p ~/.tenplex/training/$JOBID

for host in $HOSTS; do
	scp -r $host:~/.tenplex/training/$JOBID/* ~/.tenplex/training/$JOBID &
done

wait
