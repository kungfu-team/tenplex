#!/bin/sh
set -e

flags() {
    echo -detect-self-ip eth0

    # echo -reinstall
    echo -u kungfu

    echo -state-migrator /usr/bin/state-migrator
}

/usr/bin/tenplex-scheduler $(flags)

echo "$0 stopped"
