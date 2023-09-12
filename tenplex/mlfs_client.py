import copy
import json
import os
import pickle
import re

import numpy as np
import requests
import torch

from .tensor_file import TensorRequester


class MLFSClient:

    def __init__(self, req_ip: str, ctrl_port: int):
        self.ctrl_port = ctrl_port
        self.req_ip = req_ip
        self.ten_requester = TensorRequester(ctrl_port, req_ip)

        # DEBUG
        #  self.total_txt = 0
        #  self.failed_txt = []
        #  self.total_object = 0
        #  self.failed_object = []
        #  self.tensor_paths = []

    def get_type(self, obj):
        mat = re.match(r'<class \'(.*)\'>', str(type(obj)))
        if mat is not None:
            return mat.group(1)
        raise ValueError("mat is None")

    def upload_txt(self, path, txt):

        data = bytes(txt, 'utf-8')

        headers = {
            'Content-type': 'text/plain',
            'x-replace': 'true',
        }
        endpoint = f'http://{self.req_ip}:{self.ctrl_port}/upload?path={path}'
        r = requests.post(endpoint, headers=headers, data=data)
        r.raise_for_status()

        #  self.total_txt += 1

    def upload_object(self, path, obj, typ=None):
        if typ is None:
            typ = self.get_type(obj)
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
            raise ValueError(
                f"ERROR: type {typ} not supported for upload object")

        headers = {
            'Content-type': typ,
        }
        endpoint = f'http://{self.req_ip}:{self.ctrl_port}/upload?path={path}'
        r = requests.post(endpoint, headers=headers, data=data)
        if r.status_code != 200:
            r.raise_for_status()
            #  self.failed_object.append(path)

        #  self.total_object += 1

    def add_values(self, vals, new_vals):
        if isinstance(new_vals, list):
            vals.extend(new_vals)
        if isinstance(new_vals, tuple):
            vals.append(new_vals)
        return vals

    def upload_dir_meta(self, value, keys, base_path):
        length = len(value)
        metadata = f'list\n{length}'
        str_keys = [str(k) for k in keys]
        str_keys.append('dir')
        keys_path = '/'.join(str_keys)
        txt_path = os.path.join(base_path, keys_path)
        txt_path = txt_path + '.meta'
        self.upload_txt(txt_path, metadata)

    def save_traverse(self, value, base_path: str, keys=None):
        if keys is None:
            keys = []

        if isinstance(value, dict):
            for key, val in value.items():
                new_keys = copy.deepcopy(keys)
                new_keys.append(key)
                self.save_traverse(val, base_path, new_keys)
            return
        if isinstance(value, (list, set, tuple)):
            self.upload_dir_meta(value, keys, base_path)

            for i, val in enumerate(value):
                new_keys = copy.deepcopy(keys)
                new_keys.append(f'{i}')
                self.save_traverse(val, base_path, new_keys)
            return

        str_keys = [str(k) for k in keys]
        keys_path = '/'.join(str_keys)
        keys_path = os.path.join(base_path, keys_path)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().numpy()
            typ = self.get_type(tensor)
            tensor_path = keys_path + f'.{typ}'
            #  if tensor_path in self.tensor_paths:
            #      print(f"torch {tensor_path} already uploaded")
            #  self.tensor_paths.append(tensor_path)
            self.ten_requester.upload_tensor(tensor_path, tensor)
            return
        if isinstance(value, np.ndarray):
            print(f'{keys} is np.ndarray')
            typ = self.get_type(value)
            tensor_path = keys_path + f'.{typ}'
            #  if tensor_path in self.tensor_paths:
            #      print(f"np {tensor_path} already uploaded")
            #  self.tensor_paths.append(tensor_path)
            self.ten_requester.upload_tensor(tensor_path, value)
            return

        if value is None:
            value = 'None'
            self.upload_object(keys_path, value, 'none')
            return

        self.upload_object(keys_path, value)

    def get(self, path, ctype):
        url = f'http://{self.req_ip}:{self.ctrl_port}/query'
        res = requests.get(url,
                           headers={"Content-Type": ctype},
                           params={"path": path})
        res.raise_for_status()
        return res

    def get_dir(self, path):
        res = self.get(path, "x-dir")
        stru = res.text.split("\n")
        stru.sort()
        return stru

    def get_tensor(self, path):
        res = self.get(path, "x-tensor")
        tensor_data = res.content
        head = res.headers
        dtype = head["x-tensor-dtype"]
        dims = head["x-tensor-dims"]
        dims = [int(x) for x in dims.split(",")]
        return tensor_data, dtype, dims

    def get_text(self, path):
        res = self.get(path, "text")
        return res.text

    def get_file(self, path):
        res = self.get(path, "x-file")
        return res.content

    def delete(self, path: str) -> (int, int):
        endpoint = f"http://{self.req_ip}:{self.ctrl_port}/delete"
        params = {"path": path}
        r = requests.post(endpoint, params=params)
        r.raise_for_status()
        vals = r.content.decode("utf-8").strip().split(" ")
        num_files, num_dir = vals
        return int(num_files), int(num_dir)
