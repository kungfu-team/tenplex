#!/bin/sh
set -e

flags() {
    echo -detect-self-ip eth0

    # echo -reinstall
    echo -u kungfu

    echo -tenplex-state-transformer /usr/bin/tenplex-state-transformer
}

/usr/bin/tenplex-scheduler $(flags)

echo "$0 stopped"
