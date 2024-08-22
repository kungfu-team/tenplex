#!/bin/sh
set -e

now_sec() {
    date +%s
}

_show_duration() {
    echo "$1s"
}

measure() {
    echo "BEGIN $@"
    local begin=$(now_sec)
    $@
    local end=$(now_sec)
    local duration=$((end - begin))
    echo "END $@, took $(_show_duration $duration)" | tee -a measure.log
}

upgrade_cluster() {
    measure ansible-playbook -i hosts.txt ./tenplex.yml

    measure ansible-playbook -i hosts.txt ./tenplex-2.yml
}

measure
