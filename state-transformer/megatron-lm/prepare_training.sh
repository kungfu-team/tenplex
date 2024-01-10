#!/bin/bash
set -e

rm -r /data/marcel/cont_pp2_mp2/*/ckpt/iter_0000200/*

for i in {0..3}
do
    cp -r /data/marcel/transformed/* /data/marcel/cont_pp2_mp2/${i}/ckpt/iter_0000200
done

cp -r /data/marcel/cont_pp2_mp2 /data/marcel/training
