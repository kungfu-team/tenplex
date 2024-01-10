import os

import numpy as np
import torch


def main():
    param_name = 'weight'
    layer_number = 0
    source_mp_degree = 4
    target_mp_degree = 2
    source_dir = '/data/marcel/united'
    target_dir = '/data/marcel/target'
    united_shape = (64, 16)

    source_rank_length = united_shape[0] // source_mp_degree
    target_rank_length = united_shape[0] // target_mp_degree

    united_tensor = torch.arange(np.prod(united_shape))
    united_tensor = torch.reshape(united_tensor, united_shape)
    
    for source_mp_rank in range(source_mp_degree):
        a, b = (source_mp_rank * source_rank_length, (source_mp_rank + 1) * source_rank_length)
        tensor = united_tensor[a:b]
        ckpt = {param_name: tensor}
        ckpt_path = os.path.join(source_dir,
                f'layer_{layer_number:02d}-model_{source_mp_rank:02d}-model_states.pt')
        torch.save(ckpt, ckpt_path)

    for target_mp_rank in range(target_mp_degree):
        a, b = (target_mp_rank * target_rank_length, (target_mp_rank + 1) * target_rank_length)
        tensor = united_tensor[a:b]
        ckpt = {param_name: tensor}
        ckpt_path = os.path.join(target_dir,
                f'layer_{layer_number:02d}-model_{target_mp_rank:02d}-model_states.pt')
        torch.save(ckpt, ckpt_path)


if __name__ == '__main__':
    main()
