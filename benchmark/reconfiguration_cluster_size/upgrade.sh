#!/bin/sh
set -e

ansible-playbook -i hosts.txt ./tenplex.yml

