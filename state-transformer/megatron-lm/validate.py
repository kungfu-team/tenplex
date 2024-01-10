import argparse
import copy
import glob
import os

import numpy as np
import torch


def load_ckpt(path):
    return torch.load(path, map_location=torch.device('cpu'))


def calc_max_tensor_diff(ten_a, ten_b, key_seq):
    np_ten_a = ten_a.detach().numpy()
    np_ten_b = ten_b.detach().numpy()
    abs_diff = np.abs(np_ten_a - np_ten_b)
    return abs_diff.max()


def compare_ckpts(ckpt_target, ckpt_b, key_seq=[]):
    if isinstance(ckpt_target, torch.Tensor):
        if ckpt_target.shape != ckpt_b.shape:
            print(f'{key_seq}: shape {ckpt_target.shape} != {ckpt_b.shape}')
        else:
            max_diff = calc_max_tensor_diff(ckpt_target, ckpt_b, key_seq)
            if max_diff > 0.0:
                print(f'{str(key_seq):100s} max diff {max_diff:e}')
            else:
                print(f'{str(key_seq):100s} equal, {ckpt_target.shape}')
    elif isinstance(ckpt_target, dict):
        for key, val in ckpt_target.items():
            try:
                val_b = ckpt_b[key]
            except KeyError:
                print(f'{key_seq}: {key} missing in ckpt b')
                continue
            except:
                print(type(ckpt_b))
                print(ckpt_b)
                return
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(key)
            compare_ckpts(val, val_b, new_key_seq)
    elif isinstance(ckpt_target, (list, set, tuple)):
        for i, val in enumerate(ckpt_target):
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(i)
            try:
                val_b = ckpt_b[i]
            except IndexError:
                print(f'IndexError for {new_key_seq}')
                continue
            except KeyError:
                print(f'KeyError for {new_key_seq}')
                print(f'type of key {type(i)}')
                continue
            compare_ckpts(val, val_b, new_key_seq)
    elif isinstance(ckpt_target, np.ndarray):
        if ckpt_target.shape != ckpt_b.shape:
            print(f'{key_seq}: shape {ckpt_target.shape} != {ckpt_b.shape}')
    elif isinstance(ckpt_target, argparse.Namespace):
        for pair_a, pair_b in zip(
                vars(ckpt_target).items(),
                vars(ckpt_b).items()):
            key_a, val_a = pair_a
            _, val_b = pair_b
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(key_a)
            compare_ckpts(val_a, val_b, new_key_seq)
    else:
        if ckpt_target != ckpt_b:
            print(f'{key_seq}: {ckpt_target} and {ckpt_b} are unequal')


def main():
    target_dir = '/data/marcel/target'
    transformed_dir = '/data/marcel/transformed'

    #  transformed_layer_ckpt_paths = glob.glob(transformed_dir + '/layer*')
    #  transformed_layer_ckpt_paths.sort()
    #  for transformed_layer_ckpt_path in transformed_layer_ckpt_paths:
    #      transformed_layer = torch.load(transformed_layer_ckpt_path, map_location='cpu')
    #      transformed_layer_name = os.path.basename(transformed_layer_ckpt_path)
    #      target_layer_ckpt_path = os.path.join(target_dir, transformed_layer_name)
    #      target_layer = torch.load(target_layer_ckpt_path, map_location='cpu')
    #      print(transformed_layer_name)
    #      compare_ckpts(target_layer, transformed_layer)

    transformed_state_paths = glob.glob(transformed_dir + '/mp*')
    transformed_state_paths.sort()
    for transformed_state_path in transformed_state_paths:
        print(f'transformed_state_path {transformed_state_path}')
        transformed_state = torch.load(transformed_state_path, map_location='cpu')
        transformed_state_name = os.path.basename(transformed_state_path)
        target_state_path = os.path.join(target_dir, transformed_state_name)
        print(f'target_state_path {target_state_path}')
        target_state = torch.load(target_state_path, map_location='cpu')
        print(transformed_state_name)
        compare_ckpts(target_state, transformed_state)


if __name__ == '__main__':
    main()
