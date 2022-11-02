import argparse
import copy
import os
import pickle
import re

import numpy as np
import requests
import torch

from tensor_file import query_tensor_file, read_tensor_file, upload_tensor


def get_type(obj):
    mat = re.match(r'<class \'(.*)\'>', str(type(obj)))
    return mat.group(1)


def upload_txt(path, txt):
    host = '127.0.0.1'
    ctrl_port = 20010

    data = bytes(txt, 'utf-8')

    headers = {
        'Content-type': 'text/plain',
    }
    endpoint = f'http://{host}:{ctrl_port}/upload?path={path}'
    r = requests.post(endpoint, headers=headers, data=data)
    assert (r.status_code == 200)


def upload_object(path, obj, typ=None):
    host = '127.0.0.1'
    ctrl_port = 20010

    if typ is None:
        typ = get_type(obj)
    path = path + f'.{typ}'
    data = pickle.dumps(obj)

    headers = {
        'Content-type': typ,
    }
    endpoint = f'http://{host}:{ctrl_port}/upload?path={path}'
    r = requests.post(endpoint, headers=headers, data=data)
    assert (r.status_code == 200)


def add_values(vals, new_vals):
    if isinstance(new_vals, list):
        vals.extend(new_vals)
    if isinstance(new_vals, tuple):
        vals.append(new_vals)
    return vals


def traverse_ckpt(base_path: str, value, keys=None):
    if keys is None:
        keys = []

    if isinstance(value, dict):
        for key, val in value.items():
            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            traverse_ckpt(base_path, val, new_keys)
        return
    if isinstance(value, (list, set, tuple)):
        length = len(value)
        metadata = f'list\n{length}'
        str_keys = [str(k) for k in keys]
        str_keys.append('dir')
        key_path = '/'.join(str_keys)
        txt_path = key_path + '.meta'
        txt_path = os.path.join(base_path, txt_path)
        upload_txt(txt_path, metadata)

        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(f'{i}')
            traverse_ckpt(base_path, val, new_keys)
        return

    str_keys = [str(k) for k in keys]
    key_path = '/'.join(str_keys)
    key_path = os.path.join(base_path, key_path)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().numpy()
        typ = get_type(tensor)
        tensor_path = key_path + f'.{typ}'
        upload_tensor(tensor_path, tensor)
        return
    if isinstance(value, np.ndarray):
        print(f'{keys} is np.ndarray')
        typ = get_type(value)
        tensor_path = key_path + f'.{typ}'
        upload_tensor(tensor_path, value)
        return

    if value is None:
        value = 'None'
        upload_object(key_path, value, 'none')
        return

    upload_object(key_path, value)


def write_ckpt(args):
    size = args.size
    step = args.step
    timestamp = args.timestamp
    # Megatron-LM
    for i in range(size):
        base_path = os.path.join(args.ckpt_path, f'{i}/ckpt/iter_{step:07d}')
        for entry in os.scandir(base_path):
            if entry.is_dir():
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_file():
                        ckpt = torch.load(sub_entry.path, map_location='cpu')
                        #  ckpt_base_path = os.path.join(args.mlfs_path,
                        #                                f'{i}/{entry.name}')
                        ckpt_base_path = f'{timestamp}/{i}/{entry.name}'
                        traverse_ckpt(ckpt_base_path, ckpt)


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--mlfs-path', type=str)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--size', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--timestamp', type=str)
    args = parser.parse_args()

    write_ckpt(args)


if __name__ == '__main__':
    main()
