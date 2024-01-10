import copy
import os

import torch


def load_ckpt(path):
    return torch.load(path, map_location=torch.device('cpu'))


def compare_values(ten_out, ten_in, key_seq):
    #  print(f'compare {key_seq}')
    #  print(f'ten_out shape {ten_out.shape}')
    #  print(f'ten_in shape {ten_in.shape}')
    length = ten_in.shape[0]
    #  print(f'{key_seq} ten_in shape {ten_in.shape}')
    #  print(f'length {length}')
    for i in range(length):
        a = ten_out[i]
        b = ten_in[i]
        #  print(f'a {a} =? {b} b')
        if a != b:
            print(f'{key_seq}: {i}th value of {length} values is unequal')
            #  return


def compare_ckpts(ckpt_out, ckpt_in, key_seq=[]):
    if isinstance(ckpt_out, dict):
        for key, val in ckpt_out.items():
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(key)
            try:
                val_in = ckpt_in[key]
            except KeyError:
                print(f'{new_key_seq}: {key} missing in ckpt_in')
                continue
            compare_ckpts(val, val_in, new_key_seq)
    elif isinstance(ckpt_out, (list, tuple)):
        for i, val in enumerate(ckpt_out):
            new_key_seq = copy.deepcopy(key_seq)
            new_key_seq.append(i)
            try:
                val_in = ckpt_in[i]
            except IndexError:
                print(f'IndexError for {new_key_seq}')
                continue
            compare_ckpts(val, val_in, new_key_seq)
    elif isinstance(ckpt_out, torch.Tensor):
        compare_values(ckpt_out, ckpt_in, key_seq)


def compare_2to1():
    global_step = 2000
    dir_in = f'/data/marcel/deepspeed/mp2/global_step{global_step}'
    dir_out = f'/data/marcel/repartition/2-to-1/global_step{global_step}'

    # in 0
    rank = 0
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(dir_in, model_ckpt_name)
    model_in0 = load_ckpt(model_ckpt_path)
    zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'
    zero_ckpt_path = os.path.join(dir_in, zero_ckpt_name)
    zero_in0 = load_ckpt(zero_ckpt_path)

    # in 1
    rank = 1
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(dir_in, model_ckpt_name)
    model_in1 = load_ckpt(model_ckpt_path)
    zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'
    zero_ckpt_path = os.path.join(dir_in, zero_ckpt_name)
    zero_in1 = load_ckpt(zero_ckpt_path)

    # out
    rank = 0
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(dir_out, model_ckpt_name)
    model_out = load_ckpt(model_ckpt_path)
    zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'
    zero_ckpt_path = os.path.join(dir_out, zero_ckpt_name)
    zero_out = load_ckpt(zero_ckpt_path)

    #  compare_ckpts(zero_out, zero_in0)
    compare_ckpts(model_in1['torch_rng_state'], model_in0['torch_rng_state'], ['torch_rng_state'])
    compare_ckpts(model_in1['rng_tracker_states']['model-parallel-rng'],
                  model_in0['rng_tracker_states']['model-parallel-rng'],
                  ['rng_tracker_states', 'model-parallel-rng'])


def main():
    print('values')
    compare_2to1()


if __name__ == '__main__':
    main()
