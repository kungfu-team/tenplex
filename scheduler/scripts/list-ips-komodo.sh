#!/bin/sh
set -e

echo "10.10.10.1"
echo "10.10.10.2"
echo "10.10.10.3"
echo "10.10.10.4"

# cd $(dirname $0)/..

# for i in $(seq 4); do
#     if [ $i -eq 1 ]; then # hack
#         ip -o -4 addr list eth0 | awk -F ' *|/' '{print $4}'
#     else
#         domain=komodo$(printf "%02d" $i).doc.res.ic.ac.uk
#         host $domain | awk '{print $4}'
#     fi
# done
