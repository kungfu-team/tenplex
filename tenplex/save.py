import argparse
import copy
import os
import pickle
import re

import numpy as np
import requests
import torch

from .tensor_file import TensorRequester


class MLFSClient:

    def __init__(self, ctrl_port, req_ip):
        self.ctrl_port = ctrl_port
        self.req_ip = req_ip
        self.ten_requester = TensorRequester(ctrl_port, req_ip)

        # DEBUG
        #  self.total_txt = 0
        #  self.failed_txt = []
        #  self.total_object = 0
        #  self.failed_object = []

    def get_type(self, obj):
        mat = re.match(r'<class \'(.*)\'>', str(type(obj)))
        return mat.group(1)

    def upload_txt(self, path, txt):

        data = bytes(txt, 'utf-8')

        headers = {
            'Content-type': 'text/plain',
            'x-replace': 'true',
        }
        endpoint = f'http://{self.req_ip}:{self.ctrl_port}/upload?path={path}'
        r = requests.post(endpoint, headers=headers, data=data)
        if r.status_code != 200:
            r.raise_for_status()
            #  self.failed_txt.append(path)

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
            self.ten_requester.upload_tensor(tensor_path, tensor)
            return
        if isinstance(value, np.ndarray):
            print(f'{keys} is np.ndarray')
            typ = self.get_type(value)
            tensor_path = keys_path + f'.{typ}'
            self.ten_requester.upload_tensor(tensor_path, value)
            return

        if value is None:
            value = 'None'
            self.upload_object(keys_path, value, 'none')
            return

        self.upload_object(keys_path, value)


def save(ckpt: dict, job_id: str, step: int, device_rank: int, ip: str):
    # DEBUG
    #  print("PRINT STACK")
    #  traceback.print_stack(file=sys.stdout)
    #  print("PRINT STACK finished")

    mlfs_path = "/data/mlfs"  # TODO: make an argument
    save_path = os.path.join(mlfs_path, f"save{step}")
    if os.path.exists(save_path):
        print("SAVER save directory already exists")
        return

    client = MLFSClient(20010, ip)
    client.save_traverse(
        ckpt,
        os.path.join(f"job/{job_id}",
                     os.path.join(f"save{step}", str(device_rank))))
    client.upload_txt(f"job/{job_id}/iter", str(step))

    # DEBUG
    #  print(f"SAVER object total {client.total_object},"
    #        f" failed {len(client.failed_object)}")
    #  for fail in client.failed_object:
    #      print(f"SAVER object failed {fail}")
    #  print(f"SAVER txt total {client.total_txt},"
    #        f" failed {len(client.failed_txt)}")
    #  for fail in client.failed_txt:
    #      print(f"SAVER txt failed {fail}")
    #  print(f"SAVER tensor total {client.ten_requester.total_upload},"
    #        f"failed {len(client.ten_requester.failed_upload)}")
    #  for fail in client.ten_requester.failed_upload:
    #      print(f"SAVER tensor failed {fail}")


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
