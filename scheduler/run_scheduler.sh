#!/bin/bash
set -e

echo "Building scheduler ..."
make

echo "Running scheduler ..."
PREFIX=$HOME/.tenplex/scheduler

flag() {
    # echo -detect-self-ip eth0
    echo -detect-self-ip ib0
    echo -reinstall
    echo -u marcel
    echo -state-migrator /home/marcel/Elasticity/Repo/state-migrator/go/bin/state-migrator
}

if [ ! -d transformer-checkpoint ]; then
    git clone git@github.com:/kungfu-team/transformer-checkpoint.git
fi

cd transformer-checkpoint
git pull
cd -

$PREFIX/bin/tenplex-scheduler $(flag)

echo "$0 done"
