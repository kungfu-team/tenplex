import argparse
import copy
import os

import numpy as np
import torch

from tenplex.load import load


def traverse(value, keys=None):
    if keys is None:
        keys = []

    if isinstance(value, dict):
        for key, val in value.items():
            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            traverse(val, new_keys)
        return
    if isinstance(value, (list, set, tuple)):
        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(f'{i}')
            traverse(val, new_keys)
        return

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().numpy()
        typ = type(tensor)
        print(f'{keys} is {typ} and shape {value.shape}')
        return
    if isinstance(value, np.ndarray):
        typ = type(value)
        print(f'{keys} is {typ} and shape {value.shape}')
        return


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--device-rank', type=int)
    parser.add_argument('--mlfs-path', type=str)
    args = parser.parse_args()

    ckpt = load(args.device_rank, args.mlfs_path)
    traverse(ckpt)


if __name__ == '__main__':
    main()
