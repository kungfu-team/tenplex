import glob
import os
import pickle
import time

import numpy as np
import torch

import tenplex
from tenplex.mlfs_client import MLFSClient
from tenplex.tensor_file import read_tensor_file


def trim_last_ext(c):
    # 'a/b/c.x.y.z' => 'a/b/c.x.y'
    a, _ = os.path.splitext(c)
    return a


def parse_value(value_str: str, name: str):
    file_ext = name.split(".")[-1]
    if file_ext == "none":
        return None
    if file_ext == "str":
        if isinstance(value_str, bytes):
            return value_str.decode("utf-8")
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

    print(f"path {path} is not a directory nor a file")


def load(device_rank: int, mlfs_path: str):
    load_start = time.time()
    pa = os.path.join(mlfs_path, "iter")
    with open(pa, "r") as iter_file:
        step = int(iter_file.read().strip())
    pa = os.path.join(mlfs_path, f"load/{device_rank}")
    print(f"load checkpoint from {pa} at step {step}")

    start_load_trav = time.time()
    ckpt = load_traverse(pa)
    load_trav_dura = time.time() - start_load_trav
    print(f"load traverse took {load_trav_dura}")

    # Megatron-LM
    ckpt['rng_state'][0]['random_rng_state'][1] = tuple(
        ckpt['rng_state'][0]['random_rng_state'][1])
    ckpt['rng_state'][0]['random_rng_state'] = tuple(
        ckpt['rng_state'][0]['random_rng_state'])

    print(
        f"loaded checkpoint from {pa} and it took {time.time() - load_start}")

    return ckpt, step


def try_int(dic, key, val):
    try:
        dic[int(key)] = val
    except ValueError:
        dic[key] = val
    return dic


def set_value_try(ckpt, keys, name, value):
    ele = ckpt

    for key in keys:
        if key not in ele:
            try_int(ele, key, {})

        try:
            ele = ele[int(key)]
        except ValueError:
            ele = ele[key]

    try_int(ele, name, value)

    return ckpt


def set_value(ckpt, keys, name, value):
    ele = ckpt

    for key in keys:
        if key not in ele:
            ele[key] = {}
        ele = ele[key]

    ele[name] = value

    return ckpt


def get_value(ckpt, keys):
    ele = ckpt
    for key in keys:
        ele = ele[key]
    return ele


def dict_to_list(dic):
    lis = []
    for i in range(len(dic.keys())):
        lis.append(dic[i])
    return lis


def dicts_to_lists(ckpt, dir_metas):
    for met in dir_metas:
        keys = met.split("/")
        keys = keys_to_int(keys)
        keys = keys[:len(keys) - 1]
        try:
            last_key = int(keys[-1])
        except ValueError:
            last_key = keys[-1]
        parent_val = get_value(ckpt, keys[:len(keys) - 1])
        try:
            parent_val[last_key] = dict_to_list(parent_val[last_key])
        except:
            print(f"ERROR dict_to_list failed for {keys} {last_key}")

    return ckpt


def keys_to_int(keys):
    new_keys = keys
    for i, k in enumerate(keys):
        try:
            new_keys[i] = int(k)
        except ValueError:
            pass
    return new_keys


def to_int(val):
    try:
        return int(val)
    except ValueError:
        return val


def load_http(job_id: str, device_rank: int, ip: str, port: int):
    client = MLFSClient(ip, port)
    step = int(client.get_text(f"/job/{job_id}/iter"))
    base_path = f"/job/{job_id}/load/{device_rank}"
    struct = client.get_dir(base_path)
    struct_no_meta = list(filter(lambda x: not x.endswith(".meta"), struct))
    dir_meta = list(filter(lambda x: x.endswith("dir.meta"), struct))
    dir_meta.sort()

    ckpt = {}
    for ele in struct_no_meta:
        rel_path = os.path.relpath(ele, base_path)
        all_keys = rel_path.split("/")
        all_keys = keys_to_int(all_keys)
        file_name = all_keys[-1]
        keys = all_keys[:len(all_keys) - 1]

        try:
            file_name.endswith('.numpy.ndarray')
        except:
            print(f"endswith failed for {ele} and keys {keys}")

        if file_name.endswith('.numpy.ndarray'):
            name = file_name.removesuffix('.numpy.ndarray')
            name = to_int(name)
            tensor_data, dtype, dims = client.get_tensor(ele)
            typ = tenplex.tensor_file._dtypes[dtype]
            np_tensor = np.frombuffer(tensor_data, dtype=typ).reshape(dims)
            if 'np_rng_state' in ele:  # needs to stay numpy array
                ckpt = set_value(ckpt, keys, name, np_tensor)
                continue

            torch_tensor = torch.from_numpy(np_tensor)
            ckpt = set_value(ckpt, keys, name, torch_tensor)
            continue

        path_no_ext = trim_last_ext(ele)
        name = trim_last_ext(file_name)
        name = to_int(name)

        if file_name.endswith(".argparse.Namespace"):
            fil = client.get_file(path_no_ext)
            obj = pickle.loads(fil)
            ckpt = set_value(ckpt, keys, name, obj)
            continue

        fil = client.get_file(path_no_ext)
        val = parse_value(fil, file_name)
        ckpt = set_value(ckpt, keys, name, val)

    dir_meta = [os.path.relpath(x, base_path) for x in dir_meta]
    dir_meta.sort()
    ckpt = dicts_to_lists(ckpt, dir_meta)

    # Megatron-LM
    ckpt['rng_state'][0]['random_rng_state'][1] = tuple(
        ckpt['rng_state'][0]['random_rng_state'][1])
    ckpt['rng_state'][0]['random_rng_state'] = tuple(
        ckpt['rng_state'][0]['random_rng_state'])

    return ckpt, step
