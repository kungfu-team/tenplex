#!/bin/sh

set -e

if [ -z "$SAS" ]; then
    echo "missing SAS"
    exit 1
fi

prefix=https://kungfu.blob.core.windows.net/testdata

download() {
    local file=$1
    URL="$prefix/$file?$SAS"
    echo $URL
    wget -O data/$file $URL
}

mkdir -p data/global_step5

download global_step5/mp_rank_00_model_states.pt
# download global_step5/zero_pp_rank_0_mp_rank_00_optim_states.pt
