import json
import os

import numpy as np
import torch


def load_ckpt(path):
    return torch.load(path, map_location=torch.device('cpu'))


def main():
    global_step = 10
    mp_size = 4
    direc = f'/data/marcel/ckpt_mp{mp_size}/0/global_step{global_step}'
    rank = 0

    model_ckpt_name = f'mp_rank_{rank:02d}_model_states.pt'
    model_ckpt_path = os.path.join(direc, model_ckpt_name)
    model_ckpt = load_ckpt(model_ckpt_path)

    with open(f'shapes_mp{mp_size}.json', 'w') as json_file:
        json.dump(model_ckpt['param_shapes'], json_file, indent=4)

    #  zero_ckpt_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'
    #  zero_ckpt_path = os.path.join(direc, zero_ckpt_name)
    #  zero_ckpt = load_ckpt(zero_ckpt_path)


if __name__ == '__main__':
    main()
