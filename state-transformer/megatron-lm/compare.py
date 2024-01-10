import os

import numpy as np
import torch


def calc_max_tensor_diff(ten_a, ten_b):
    np_ten_a = ten_a.numpy()
    np_ten_b = ten_b.numpy()
    abs_diff = np.abs(np_ten_a - np_ten_b)
    return abs_diff.max()


def compare_tensors(target, source, target_mp_rank, source_mp_rank):
    target_shape = target.size()
    source_shape = source.size()
    if target_shape[0] != source_shape[0]:
        print('first dimension different')
    for target_row in range(target_shape[0]):
        for source_row in range(source_shape[0]):
            if calc_max_tensor_diff(target[target_row], source[source_row]) < 1e-2:
                print(f'target mp rank {target_mp_rank}'
                      f' target row {target_row}'
                      f' at source mp rank {source_mp_rank},'
                      f' row {source_row} < 1e-2')
                return
        print(f'target mp rank {target_mp_rank}'
              f' target row {target_row}'
              f' at source mp rank {source_mp_rank} not found')


def main():
    #  param_name = 'word_embeddings.weight'
    #  layer_number = 0
    param_name = 'mlp.dense_h_to_4h.weight'
    layer_number = 2
    target_path = '/data/marcel/target'
    source_path = '/data/marcel/united'
    target_mp_degree = 2
    source_mp_degree = 4

    for target_mp_rank in range(target_mp_degree):
        target_layer_name = f'layer_{layer_number:02d}-model_{target_mp_rank:02d}-model_states.pt'
        target_layer_path = os.path.join(target_path, target_layer_name)
        target_layer = torch.load(target_layer_path, map_location='cpu')
        target_tensor = target_layer[param_name]

        for source_mp_rank in range(source_mp_degree):
            source_layer_name = f'layer_{layer_number:02d}-model_{source_mp_rank:02d}-model_states.pt'
            source_layer_path = os.path.join(source_path, source_layer_name)
            source_layer = torch.load(source_layer_path, map_location='cpu')
            source_tensor = source_layer[param_name]

            compare_tensors(target_tensor, source_tensor, target_mp_rank, source_mp_rank)


if __name__ == '__main__':
    main()
