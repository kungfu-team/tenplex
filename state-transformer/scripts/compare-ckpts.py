import argparse
import copy
import os

import numpy as np
import torch

from hist import cmp_hist, hist


def load_ckpt(path):
    return torch.load(path, map_location=torch.device('cpu'))


def calc_max_tensor_diff(ten_a, ten_b, key_seq):
    np_ten_a = ten_a.detach().numpy()
    np_ten_b = ten_b.detach().numpy()
    abs_diff = np.abs(np_ten_a - np_ten_b)
    return abs_diff.max()


def compare_byte_histogram(ten_a, ten_b, key_seq):
    np_ten_a = ten_a.detach().numpy()
    np_ten_b = ten_b.detach().numpy()

    d = cmp_hist(hist(np_ten_a), hist(np_ten_b), str(key_seq))
    if d == 0:
        print(f'{key_seq}: identical hist')
    else:
        print(f'{key_seq}: different hist {d} values')

    return d


def compare_values(ten_out, ten_in, key_seq):
    length = ten_in.shape[0]
    for i in range(length):
        a = ten_out[i]
        b = ten_in[i]
        #  print(f'a {a} =? {b} b')
        if a != b:
            print(f'{key_seq}: {i}th value {a} != {b}')


def compare_ckpts(ckpt_target, ckpt_b, key_seq=[]):
    if ckpt_target is not None and ckpt_b is None:
        print(f'{key_seq} is None')
        return
    if isinstance(ckpt_target, dict):
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
    elif isinstance(ckpt_target, torch.Tensor):
        if ckpt_target.shape != ckpt_b.shape:
            print(f'{key_seq}: shape {ckpt_target.shape} != {ckpt_b.shape}')
        else:
            max_diff = calc_max_tensor_diff(ckpt_target, ckpt_b, key_seq)
            if max_diff > 0.0:
                print(f'{str(key_seq):100s} max diff {max_diff:e}')
                #  compare_byte_histogram(ckpt_target, ckpt_b, key_seq)
                #  compare_values(ckpt_target, ckpt_b, key_seq)
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


def multiple():
    global_step = 2000
    mp_size = 4
    dir_repartitioned = f'/data/marcel/repartition/4to2to4/global_step{global_step}'
    dir_target = f'/data/marcel/deepspeed/mp{mp_size}/global_step{global_step}'

    for rank in range(mp_size):
        model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
        zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'

        model_ckpt_path = os.path.join(dir_repartitioned, model_ckpt_name)
        model_ckpt_target_path = os.path.join(dir_target, model_ckpt_name)
        print(f'rank {rank}, model_ckpt_path {model_ckpt_path}')
        print(f'rank {rank}, model_ckpt_target_path {model_ckpt_target_path}')
        model_ckpt = load_ckpt(model_ckpt_path)
        model_ckpt_target = load_ckpt(model_ckpt_target_path)
        compare_ckpts(model_ckpt_target, model_ckpt)

        zero_ckpt_path = os.path.join(dir_repartitioned, zero_ckpt_name)
        zero_ckpt_target_path = os.path.join(dir_target, zero_ckpt_name)
        print(f'rank {rank}, zero_ckpt_path {zero_ckpt_path}')
        print(f'rank {rank}, zero_ckpt_target_path {zero_ckpt_target_path}')
        zero_ckpt = load_ckpt(zero_ckpt_path)
        zero_ckpt_target = load_ckpt(zero_ckpt_target_path)
        compare_ckpts(zero_ckpt_target, zero_ckpt)


def hello():
    #  global_step = 100
    base_dir = '/data/marcel'
    dir_is = os.path.join(base_dir, 'transformed_new')
    dir_target = os.path.join(base_dir, 'transformed_good')

    mp_size = 4
    for rank in range(mp_size):
        model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'

        model_ckpt_path = os.path.join(dir_is, model_ckpt_name)
        model_ckpt_target_path = os.path.join(dir_target, model_ckpt_name)
        model_ckpt = load_ckpt(model_ckpt_path)
        model_ckpt_target = load_ckpt(model_ckpt_target_path)
        compare_ckpts(model_ckpt_target, model_ckpt)

    tp_size = 2
    num_layers = 24
    for layer in range(num_layers):
        for tp_rank in range(tp_size):
            model_ckpt_name = f'layer_{layer:02d}-model_{tp_rank:02d}-model_states.pt'
            model_ckpt_path = os.path.join(dir_is, model_ckpt_name)
            model_ckpt_target_path = os.path.join(dir_target, model_ckpt_name)
            try:
                model_ckpt = load_ckpt(model_ckpt_path)
                model_ckpt_target = load_ckpt(model_ckpt_target_path)
            except FileNotFoundError:
                continue
            compare_ckpts(model_ckpt_target, model_ckpt)


def two():
    path_a = '/data/marcel/training/0/ckpt/global_step50/mp_rank_00_model_states.pt'
    path_b = '/home/marcel/Elasticity/Repo/tenplex-run/mlfs/ckpt.pt'
    ckpt_a = load_ckpt(path_a)
    ckpt_b = load_ckpt(path_b)

    compare_ckpts(ckpt_a, ckpt_b)


def main():
    two()


if __name__ == '__main__':
    main()
