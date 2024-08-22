#!/bin/sh
set -e

ansible-playbook -i hosts.txt ./tenplex.yml
ansible-playbook -i hosts.txt ./tenplex-2.yml

