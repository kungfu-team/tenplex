import argparse
import glob
import os
import pickle

import numpy as np
import torch

from tensor_file import query_tensor_file, read_tensor_file, upload_tensor


def construct_ckpt(path: str):
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
                file_list = glob.glob(os.path.join(path, f'{i}*'))
                file_list.sort()  # needs sorting for ndarray files
                file_path = file_list[0]
                ele = construct_ckpt(file_path)
                lis.append(ele)

            return lis

        ckpt = {}
        for entry in os.scandir(path):
            name_split = entry.name.split('.', 1)
            name = name_split[0]
            try:
                key = int(name)
            except ValueError:
                key = name
            ckpt[key] = construct_ckpt(entry.path)

        return ckpt

    if os.path.isfile(path):
        name = os.path.basename(path)
        if name == 't0.txt':
            return
        if name.endswith('.meta'):
            return

        if name.endswith('.numpy.ndarray'):
            tensor = read_tensor_file(path)
            if 'np_rng_state' not in path:  # needs to stay numpy array
                torch_tensor = torch.from_numpy(tensor)
                return torch_tensor

            return tensor

        with open(path, 'rb') as fil:
            byts = fil.read()

        value = pickle.loads(byts)
        if value == 'None':
            value = None
        return value


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--mlfs-path', type=str)
    args = parser.parse_args()

    ckpt = construct_ckpt(args.mlfs_path)

    torch.save(ckpt, './ckpt.pt')

    #  import pprint
    #  pprint.pprint(ckpt)


if __name__ == '__main__':
    main()
