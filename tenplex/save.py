import argparse
import copy
import os
import pickle
import re

import numpy as np
import requests
import torch

from .tensor_file import query_tensor_file, upload_tensor


def get_type(obj):
    mat = re.match(r'<class \'(.*)\'>', str(type(obj)))
    return mat.group(1)


def upload_txt(path, txt):
    host = 'localhost'
    ctrl_port = 20010

    data = bytes(txt, 'utf-8')

    headers = {
        'Content-type': 'text/plain',
    }
    endpoint = f'http://{host}:{ctrl_port}/upload?path={path}'
    r = requests.post(endpoint, headers=headers, data=data)
    assert (r.status_code == 200)


def upload_object(path, obj, typ=None):
    host = 'localhost'
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


def save_traverse(value, base_path: str, keys=None):
    if keys is None:
        keys = []

    if isinstance(value, dict):
        for key, val in value.items():
            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            save_traverse(val, base_path, new_keys)
        return
    if isinstance(value, (list, set, tuple)):
        length = len(value)
        metadata = f'list\n{length}'
        str_keys = [str(k) for k in keys]
        str_keys.append('dir')
        keys_path = '/'.join(str_keys)
        txt_path = os.path.join(base_path, keys_path)
        txt_path = txt_path + '.meta'
        upload_txt(txt_path, metadata)

        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(f'{i}')
            save_traverse(val, base_path, new_keys)
        return

    str_keys = [str(k) for k in keys]
    keys_path = '/'.join(str_keys)
    keys_path = os.path.join(base_path, keys_path)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().numpy()
        typ = get_type(tensor)
        tensor_path = keys_path + f'.{typ}'
        upload_tensor(tensor_path, tensor)
        return
    if isinstance(value, np.ndarray):
        print(f'{keys} is np.ndarray')
        typ = get_type(value)
        tensor_path = keys_path + f'.{typ}'
        upload_tensor(tensor_path, value)
        return

    if value is None:
        value = 'None'
        upload_object(keys_path, value, 'none')
        return

    upload_object(keys_path, value)


def save(ckpt: dict, jobID: str, step: int, device_rank: int):
    save_traverse(
        ckpt,
        os.path.join(f"job/{jobID}",
                     os.path.join(f"save{step}", str(device_rank))))
    upload_txt(f"job/{jobID}/iter", str(step))


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--step', type=str)
    parser.add_argument('--device-rank', type=int)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    save(ckpt, args.step, args.device_rank)


if __name__ == '__main__':
    main()
