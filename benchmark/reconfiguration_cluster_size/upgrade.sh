#!/bin/sh
set -e

ansible-playbook -i hosts.txt ./tenplex.yml

echo "pulling image"
ansible-playbook -i hosts.txt ./tenplex-2.yml
