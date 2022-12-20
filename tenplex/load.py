import argparse
import glob
import os
import pickle

import torch

from .tensor_file import read_tensor_file


def parse_value(value_str: str, name: str):
    file_ext = name.split(".")[-1]
    if file_ext == "none":
        return None
    if file_ext == "str":
        return value_str
    if file_ext == "int":
        return int(value_str)
    if file_ext == "float":
        return float(value_str)
    if file_ext == "bool":
        return bool(value_str)

    raise ValueError(f"ERROR: type {file_ext} not supported in parse value")


def load_traverse(path: str):
    if os.path.isdir(path):
        metadata_path = os.path.join(path, 'dir.meta')
        if os.path.exists(metadata_path):
            # dir has metadata
            # dir is list
            with open(metadata_path, 'r') as meta_fil:
                metadata = meta_fil.readlines()
            length = int(metadata[1])
            if length == 0:
                return []

            lis = []
            for i in range(length):
                glob_path = os.path.join(path, f'{i}*')
                file_list = glob.glob(glob_path)
                file_list.sort()  # needs sorting for ndarray files
                if len(file_list) == 0:
                    raise ValueError(
                        f"ERROR: glob list is empty for {glob_path}")
                file_path = file_list[0]
                if file_path.endswith('.meta'):
                    continue
                ele = load_traverse(file_path)
                lis.append(ele)

            return lis

        ckpt = {}
        for entry in os.scandir(path):
            if entry.name.endswith('.meta'):
                continue

            if entry.name.endswith('.numpy.ndarray'):
                name_split = entry.name.split('.')
                name = '.'.join(name_split[0:-2])
            else:
                name_split = entry.name.split('.', 1)
                name = name_split[0]

            try:
                int_key = int(name)
                ckpt[int_key] = load_traverse(entry.path)
            except ValueError:
                ckpt[name] = load_traverse(entry.path)

        return ckpt

    if os.path.isfile(path):
        name = os.path.basename(path)
        if name == 't0.txt':
            return None

        if name.endswith('.numpy.ndarray'):
            tensor = read_tensor_file(path)
            if 'np_rng_state' in path:  # needs to stay numpy array
                return tensor

            torch_tensor = torch.from_numpy(tensor)
            return torch_tensor

        if name.endswith(".argparse.Namespace"):
            with open(path, "rb") as fil:
                return pickle.load(fil)

        with open(path, "r") as fil:
            payload = fil.read()
        return parse_value(payload, name)


def load(device_rank: int, mlfs_path: str):
    pa = os.path.join(mlfs_path, "iter")
    with open(pa, "r") as iter_file:
        step = int(iter_file.read().strip())
    pa = os.path.join(mlfs_path, f"load{step}/{device_rank}")
    ckpt = load_traverse(pa)

    # Megatron-LM
    ckpt['rng_state'][0]['random_rng_state'][1] = tuple(
        ckpt['rng_state'][0]['random_rng_state'][1])
    ckpt['rng_state'][0]['random_rng_state'] = tuple(
        ckpt['rng_state'][0]['random_rng_state'])

    return ckpt, step


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--device-rank', type=int)
    parser.add_argument('--mlfs-path', type=str)
    args = parser.parse_args()

    load(args.device_rank, args.mlfs_path)


if __name__ == '__main__':
    main()
