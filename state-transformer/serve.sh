#!/bin/bash
set -e

serve_tensor_flags=(
    --port 21234
    --data /data/marcel/with_controller/merged
)
    # --data /data/lg/ckpt/0
    # --data /data/marcel/deepspeed/mp2
    # --data /data/marcel/repartition/2-to-1

python3 ./training_state/serve_ckpt.py ${serve_tensor_flags[@]}
