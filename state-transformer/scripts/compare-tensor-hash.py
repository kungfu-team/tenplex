import collections
import copy
import hashlib
import os

import torch

from hist import cmp_hist, hist


def load_ckpt(path):
    return torch.load(path, map_location=torch.device('cpu'))


def hash_sha256(obj):
    return hashlib.sha256(obj).hexdigest()


def create_hash_list(t, dim):
    assert 0 <= dim <= 2
    hash_list = []
    for i in range(t.shape[dim]):
        if dim == 0:
            t_dim = t[i]
        else:
            t_dim = t[:, i]
        h = hash_sha256(t_dim.numpy().tobytes())
        hash_list.append(h)
    return hash_list


def compare_hashes(ten_out, ten_in_list, key_seq):
    dim = 0
    dimension_out = 0
    dimension_in = 0
    for i, (dim_out, dim_in) in enumerate(zip(ten_out.shape, ten_in_list[0].shape)):
        if dim_out != dim_in:
            dim = i
            dimension_out = dim_out
            dimension_in = dim_in
            break
    print(f'dimension_in {dimension_in}')
    print(f'dimension_out {dimension_out}')

    hash_out = create_hash_list(ten_out, dim)
    hash_in_list = [create_hash_list(ten, dim) for ten in ten_in_list]
    hash_in = []
    for hashi in hash_in_list:
        hash_in += hashi

    for k, h_in in enumerate(hash_in):
        indices_out = []
        for l, h_out in enumerate(hash_out):
            if h_in == h_out:
                indices_out.append(l)
        print(f'{key_seq}: IN hash {k} is at OUT hash {indices_out}')
    print('---+++---')


def compare_ckpts(ckpt_out, ckpt_in_list, key_seq=[]):
    if isinstance(ckpt_out, dict):
        for key, val in ckpt_out.items():
            try:
                val_in_list = [ckpt_in[key] for ckpt_in in ckpt_in_list]
            except KeyError:
                print(f'{key_seq}: {key} missing in ckpt_in_list')
                continue
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(key)
            compare_ckpts(val, val_in_list, new_key_seq)
    elif isinstance(ckpt_out, (list, tuple)):
        for i, val in enumerate(ckpt_out):
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(i)
            try:
                val_in_list = [ckpt_in[i] for ckpt_in in ckpt_in_list]
            except IndexError:
                print(f'IndexError for {new_key_seq}')
                continue
            compare_ckpts(val, val_in_list, new_key_seq)
    elif isinstance(ckpt_out, torch.Tensor):
        compare_hashes(ckpt_out, ckpt_in_list, key_seq)


def compare_2to1():
    global_step = 2000
    dir_in = f'/data/marcel/deepspeed/mp2/global_step{global_step}'
    dir_out = f'/data/marcel/repartition/2-to-1/global_step{global_step}'

    # in 0
    rank = 0
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(dir_in, model_ckpt_name)
    model_in0 = load_ckpt(model_ckpt_path)

    # in 1
    rank = 1
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(dir_in, model_ckpt_name)
    model_in1 = load_ckpt(model_ckpt_path)

    # out
    rank = 0
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(dir_out, model_ckpt_name)
    model_out = load_ckpt(model_ckpt_path)

    compare_ckpts(model_out, [model_in0, model_in1])


def main():
    compare_2to1()


if __name__ == '__main__':
    main()
