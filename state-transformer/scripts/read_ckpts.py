import json
import os
import pprint

import numpy as np
import torch


def load_ckpt(path):
    return torch.load(path, map_location=torch.device('cpu'))


def print_loss_scaler(loss_scaler):
    cur_hysteresis = loss_scaler.cur_hysteresis
    cur_iter = loss_scaler.cur_iter
    cur_scale = loss_scaler.cur_scale

    print(f'cur_hysteresis {cur_hysteresis}')
    print(f'cur_iter {cur_iter}')
    print(f'cur_scale {cur_scale}')


def save_obj_json(obj, name):
    attr_dict = {}
    for attr in dir(obj):
        value = getattr(obj, attr)
        if not callable(value):
            attr_dict[attr] = value
    with open(f'{name}.json', 'w') as json_file:
        json.dump(attr_dict, json_file, indent=4)


def something():
    global_step = 2000
    dir_repartitioned = f'/data/marcel/ckpt/global_step{global_step}'
    dir_target = f'/data/marcel/deepspeed/mp2/global_step{global_step}'
    dir_source = f'/data/marcel/deepspeed/mp4/global_step{global_step}'

    for rank in [0, 1]:
        print(f'RANK {rank}')
        model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
        zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'

        model_ckpt_path = os.path.join(dir_repartitioned, model_ckpt_name)
        model_ckpt_target_path = os.path.join(dir_target, model_ckpt_name)
        model_ckpt = load_ckpt(model_ckpt_path)
        model_ckpt_target = load_ckpt(model_ckpt_target_path)

        #  local_rank = model_ckpt_target['args'].local_rank
        #  args_rank = model_ckpt_target['args'].rank
        #  print(f'TARGET local rank {local_rank}')
        #  print(f'TARGET rank {args_rank}')
        #  local_rank = model_ckpt['args'].local_rank
        #  args_rank = model_ckpt['args'].rank
        #  print(f'REPARTITION local rank {local_rank}')
        #  print(f'REPARTITION rank {args_rank}')

        zero_ckpt_path = os.path.join(dir_repartitioned, zero_ckpt_name)
        zero_ckpt_target_path = os.path.join(dir_target, zero_ckpt_name)
        zero_ckpt_source_path = os.path.join(dir_source, zero_ckpt_name)
        zero_ckpt = load_ckpt(zero_ckpt_path)
        zero_ckpt_target = load_ckpt(zero_ckpt_target_path)
        zero_ckpt_source = load_ckpt(zero_ckpt_source_path)

        loss_scaler = zero_ckpt['optimizer_state_dict']['loss_scaler']
        save_obj_json(loss_scaler, f'repartition_{rank}')

        loss_scaler = zero_ckpt_target['optimizer_state_dict']['loss_scaler']
        save_obj_json(loss_scaler, f'target_{rank}')

        loss_scaler = zero_ckpt_source['optimizer_state_dict']['loss_scaler']
        save_obj_json(loss_scaler, f'source_{rank}')


def abc():
    global_step = 2000
    direc = f'/data/marcel/deepspeed/mp2/global_step{global_step}'
    direc_mp1 = f'/data/marcel/deepspeed/mp1/global_step{global_step}'

    rank = 0
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(direc, model_ckpt_name)
    model_ckpt = load_ckpt(model_ckpt_path)

    rank = 0
    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(direc_mp1, model_ckpt_name)
    model_mp1 = load_ckpt(model_ckpt_path)

    #  zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'
    #  zero_ckpt_path = os.path.join(direc, zero_ckpt_name)
    #  zero_ckpt = load_ckpt(zero_ckpt_path)

    param_shapes = model_ckpt['param_shapes']
    param_shapes_mp1 = model_mp1['param_shapes']

    pprint.pprint(param_shapes[0])
    pprint.pprint(param_shapes_mp1[0])


def mdp():
    pass


def main():
    mdp()


if __name__ == '__main__':
    main()
