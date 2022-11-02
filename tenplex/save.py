import argparse
import copy
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


def traverse_ckpt(value, keys=None):
    if keys is None:
        keys = []

    if isinstance(value, dict):
        for key, val in value.items():
            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            traverse_ckpt(val, new_keys)
        return
    if isinstance(value, (list, set, tuple)):
        length = len(value)
        metadata = f'list\n{length}'
        str_keys = [str(k) for k in keys]
        str_keys.append('dir')
        base_path = '/'.join(str_keys)
        txt_path = base_path + '.meta'
        upload_txt(txt_path, metadata)

        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(f'{i}')
            traverse_ckpt(val, new_keys)
        return

    str_keys = [str(k) for k in keys]
    base_path = '/'.join(str_keys)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().numpy()
        typ = get_type(tensor)
        tensor_path = base_path + f'.{typ}'
        upload_tensor(tensor_path, tensor)
        return
    if isinstance(value, np.ndarray):
        print(f'{keys} is np.ndarray')
        typ = get_type(value)
        tensor_path = base_path + f'.{typ}'
        upload_tensor(tensor_path, value)
        return

    if value is None:
        value = 'None'
        upload_object(base_path, value, 'none')
        return

    upload_object(base_path, value)


def write_ckpt(args):
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    traverse_ckpt(ckpt)


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--mlfs-path', type=str)
    parser.add_argument('--ckpt-path', type=str)
    args = parser.parse_args()

    write_ckpt(args)


if __name__ == '__main__':
    main()
