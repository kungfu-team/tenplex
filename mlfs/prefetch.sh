#!/bin/sh

set -e

make

SA=minddata
SAS=$(cat $HOME/.az/$SA.sas)

URL=https://$SA.blob.core.windows.net/data/imagenet/imagenet.md5.txt?$SAS

# wget -O imagenet.md5.txt $URL

prefetch() {
    local md5=$1
    local URL="https://minddata.blob.core.windows.net/data/imagenet/records/$2"

    ./bin/mlfs-fetch -ctrl-port 20000 -file $URL -md5 $md5
}

prefetch_idx_file() {
    local idx_file=$1
    cat $idx_file | while read line; do
        local md5=$(echo $line | awk '{print $1}')
        local filename=$(echo $line | awk '{print $2}')
        prefetch $md5 $filename
    done
}

prefetch_idx_file imagenet.md5.txt

# prefetch 8c7f3aa5f4f227f261717028d6c76c6e  train-00000-of-01024
# prefetch 99943ca2bd3c48baa633a2f4ee805f6c  train-00001-of-01024
# prefetch c117e44c7f83b80ebfbbddf990773b8a  train-00002-of-01024
# prefetch 47644a7c6c924358e207cba2f2c51727  train-00003-of-01024
# prefetch c733217f52e73fd72f6566c9569d2d40  train-00004-of-01024
# prefetch 05170c43f2c4be60b46c391d98b52481  train-00005-of-01024
# prefetch 190dbbfdd581623a1a90835bb9a23460  train-00006-of-01024
# prefetch 0663659d61497f6546e90abcf8b1e08d  train-00007-of-01024

# https://minddata.blob.core.windows.net/data/imagenet/records/train-01001-of-01024 1251
