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


def upload_txt(path, txt, ip):
    ctrl_port = 20010

    data = bytes(txt, 'utf-8')

    headers = {
        'Content-type': 'text/plain',
        'x-replace': 'true',
    }
    endpoint = f'http://{ip}:{ctrl_port}/upload?path={path}'
    r = requests.post(endpoint, headers=headers, data=data)
    if r.status_code != 200:
        r.raise_for_status()


def upload_object(path, obj, ip, typ=None):
    ctrl_port = 20010

    if typ is None:
        typ = get_type(obj)
    path = path + f'.{typ}'

    if typ == "none":
        data = "None"
    elif typ == "str":
        data = obj
    elif typ == "int":
        data = str(obj)
    elif typ == "float":
        data = str(obj)
    elif typ == "bool":
        data = str(obj)
    elif typ == "argparse.Namespace":
        data = pickle.dumps(obj)
    else:
        raise ValueError(f"ERROR: type {typ} not supported for upload object")

    headers = {
        'Content-type': typ,
    }
    endpoint = f'http://{ip}:{ctrl_port}/upload?path={path}'
    r = requests.post(endpoint, headers=headers, data=data)
    if r.status_code != 200:
        r.raise_for_status()


def add_values(vals, new_vals):
    if isinstance(new_vals, list):
        vals.extend(new_vals)
    if isinstance(new_vals, tuple):
        vals.append(new_vals)
    return vals


def upload_dir_meta(value, keys, base_path, ip):
    length = len(value)
    metadata = f'list\n{length}'
    str_keys = [str(k) for k in keys]
    str_keys.append('dir')
    keys_path = '/'.join(str_keys)
    txt_path = os.path.join(base_path, keys_path)
    txt_path = txt_path + '.meta'
    upload_txt(txt_path, metadata, ip)


def save_traverse(value, base_path: str, ip: str, keys=None):
    if keys is None:
        keys = []

    if isinstance(value, dict):
        for key, val in value.items():
            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            save_traverse(val, base_path, ip, new_keys)
        return
    if isinstance(value, (list, set, tuple)):
        upload_dir_meta(value, keys, base_path, ip)

        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(f'{i}')
            save_traverse(val, base_path, ip, new_keys)
        return

    str_keys = [str(k) for k in keys]
    keys_path = '/'.join(str_keys)
    keys_path = os.path.join(base_path, keys_path)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().numpy()
        typ = get_type(tensor)
        tensor_path = keys_path + f'.{typ}'
        upload_tensor(tensor_path, tensor, ip)
        return
    if isinstance(value, np.ndarray):
        print(f'{keys} is np.ndarray')
        typ = get_type(value)
        tensor_path = keys_path + f'.{typ}'
        upload_tensor(tensor_path, value, ip)
        return

    if value is None:
        value = 'None'
        upload_object(keys_path, value, ip, 'none')
        return

    upload_object(keys_path, value, ip)


def save(ckpt: dict, jobID: str, step: int, device_rank: int, ip: str):
    save_traverse(
        ckpt,
        os.path.join(f"job/{jobID}",
                     os.path.join(f"save{step}", str(device_rank))), ip)
    upload_txt(f"job/{jobID}/iter", str(step), ip)


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--step', type=str)
    parser.add_argument('--device-rank', type=int)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    save(ckpt, '0', args.step, args.device_rank, 'localhost')


if __name__ == '__main__':
    main()
