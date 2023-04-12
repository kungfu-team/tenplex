import os

import torch

#  import tenplex
from tenplex.load import load_traverse


def load(device_rank: int, mlfs_path: str):
    pa = os.path.join(mlfs_path, str(device_rank))
    ckpt = load_traverse(pa)
    if ckpt is None:
        raise ValueError("checkpoint is None")
    else:
        print(f"checkpoint keys {ckpt.keys()}")

    # Megatron-LM
    ckpt['rng_state'][0]['random_rng_state'][1] = tuple(
        ckpt['rng_state'][0]['random_rng_state'][1])
    ckpt['rng_state'][0]['random_rng_state'] = tuple(
        ckpt['rng_state'][0]['random_rng_state'])

    return ckpt


def main():
    mlfs_path = "/mnt/mlfs/job/5a9a0fe5e6/load50"
    ckpt = load(device_rank=0, mlfs_path=mlfs_path)
    torch.save(ckpt, "./torch.ckpt")


if __name__ == "__main__":
    main()
