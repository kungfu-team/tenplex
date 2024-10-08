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

wait_docker() {
    measure ansible-playbook -i hosts.txt ./docker.yml # took 269s
}

upgrade_cluster() {
    measure ansible-playbook -i hosts.txt ./tenplex.yml

    for i in $(seq 10); do
        wait_docker
        sleep 2
    done
}

measure upgrade_cluster
