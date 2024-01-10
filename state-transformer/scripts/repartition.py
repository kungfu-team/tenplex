import argparse
import collections
import copy
import json
import os
import pickle
import random
import re

import numpy as np
import requests
import torch


def get_tensor(path, rang=None):
    url = 'http://127.0.0.1:21234' + '/' + path
    if rang:
        payload = {'range': rang}
    else:
        payload = None
    result = requests.get(url, params=payload)
    tensor = pickle.loads(result.content)

    return tensor


def get_struct(path):
    url = 'http://127.0.0.1:21234' + '/' + path
    result = requests.get(url)
    struct = json.loads(result.content)

    return struct


def save_model_ckpt(checkpoint, path, rank):
    ckpt_path = os.path.join(path, f'mp_rank_{rank:02d}_model_states.pt')
    torch.save(checkpoint, ckpt_path)


def save_optimiser_ckpt(checkpoint, path, rank):
    file_name = f'zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt'
    ckpt_path = os.path.join(path, file_name)
    torch.save(checkpoint, ckpt_path)


def search_ckpt(ckpt, dir_path, key_seq=[]):
    def check_val(val, key, dir_path):
        new_seq = copy.deepcopy(key_seq)
        new_seq.append(key)
        if val is None:
            new_seq_str = [str(ele) for ele in new_seq]
            request_path = dir_path + '/' + '/'.join(new_seq_str)
            ckpt[key] = get_tensor(request_path)
            if ckpt[key] is None:
                print(f'{key_seq} {key} is None.')
        else:
            search_ckpt(val, dir_path, new_seq)

    if ckpt is None:
        raise TypeError('f{key_seq} is None.')
    if isinstance(ckpt, dict):
        for key, val in ckpt.items():
            check_val(val, key, dir_path)
        return
    if isinstance(ckpt, torch.Tensor):
        return
    if isinstance(ckpt, (list, set, tuple)):
        for i, val in enumerate(ckpt):
            check_val(val, i, dir_path)


def map_requests(source_dim, target_dim, target_mp_size, rank):
    """ returns the device AND lower and upper bound for a merged model """
    full_size = target_dim * target_mp_size
    (a, b) = (rank * target_dim, rank * target_dim + target_dim)
    source_mp_size = full_size // source_dim
    reqs = {}
    for device in range(source_mp_size):
        (c, d) = (device * source_dim, (device + 1) * source_dim)
        if a < c:  # before
            if b < c:  # before
                pass
            elif c < b <= d:  # within
                reqs[device] = [c, b]
            elif d < b:  # after
                reqs[device] = [c, d]
        elif c <= a < d:  # within
            if b < c:  # before
                raise ValueError(f'b={b} is smaller than a={a}')
            elif c < b <= d:  # within
                reqs[device] = [a, b]
            elif d < b:  # after
                reqs[device] = [a, d]
        elif d < a:  # after
            pass

    return reqs


def request_tensors(reqs, dim, path):
    tensors = []
    for device, r in reqs.items():
        device_path = path.format(requested_rank=device)
        if dim == 0:  # first dimension
            r_str = str(r)
            r_str = r_str.replace(',', ':')
        elif dim == 1:  # second dimenson
            r_str = f'[:,{r[0]}:{r[1]}]'
        else:
            raise NotImplementedError('Requesting tensor along dimension higher 2')
        tensor = get_tensor(device_path, r_str)
        tensors.append(tensor)

    return tensors


def source_adjust_requests(reqs, source_dim):
    for device, rang in reqs.items():
        offset_zero = rang[0] % source_dim
        if offset_zero == 0:
            lower = 0
        else:
            lower = offset_zero
        offset_one = rang[1] % source_dim
        if offset_one == 0:
            upper = source_dim
        else:
            upper = offset_one
 
        assert lower < upper

        reqs[device] = [lower, upper]

    return reqs



def repartition_tensor(source_shape, target_shape, path, target_mp_size, rank):
    requested_rank = 0  # TODO use random rank
    if source_shape == target_shape:  # replicate
        path = path.format(requested_rank=requested_rank)
        tensor = get_tensor(path)
    else:  # repartition
        for dim, (source_dim, target_dim) in enumerate(zip(source_shape, target_shape)):
            if source_dim != target_dim:
                reqs = map_requests(source_dim, target_dim, target_mp_size, rank)
                reqs = source_adjust_requests(reqs, source_dim)
                tensors = request_tensors(reqs, dim, path)
                tensor = torch.cat(tensors, dim=dim)
                break

    assert tuple(tensor.shape) == tuple(target_shape)

    return tensor


def find_key_group(shapes, shape_key):
    for group in range(len(shapes)):
        if shape_key in shapes[group]:
            return group


def repartition_model(source_shapes, target_shapes, args):
    rank = args.rank
    target_mp_size = args.target_mp_size
    global_step = args.global_step
    requested_rank = random.randint(0, args.source_mp_size - 1)
    path_dir = f'global_step{global_step}'
    path_dir += '/mp_rank_{requested_rank:02d}_model_states.pt'
    ckpt = get_struct(path_dir.format(requested_rank=requested_rank))

    # transformer
    transformer = ckpt['module']['language_model']['transformer']
    for k in transformer.keys():
        shape_key = 'language_model.transformer.' + k
        key_group = find_key_group(source_shapes, shape_key)
        source_shape = source_shapes[key_group][shape_key]
        target_shape = target_shapes[key_group][shape_key]
        path = path_dir + f'/module/language_model/transformer/{k}'
        tensor = repartition_tensor(source_shape, target_shape, path, target_mp_size, rank)
        transformer[k] = tensor

    # position embedding
    shape_key = 'language_model.embedding.position_embeddings.weight'
    key_group = find_key_group(source_shapes, shape_key)
    source_shape = source_shapes[key_group][shape_key]
    target_shape = target_shapes[key_group][shape_key]
    shape_key_slash = shape_key.replace('.', '/')
    path = path_dir + f'/module/{shape_key_slash}'
    tensor = repartition_tensor(source_shape, target_shape, path, target_mp_size, rank)
    ckpt['module']['language_model']['embedding']['position_embeddings']['weight'] = tensor

    # word embedding
    shape_key = 'language_model.embedding.word_embeddings.weight'
    key_group = find_key_group(source_shapes, shape_key)
    source_shape = source_shapes[key_group][shape_key]
    target_shape = target_shapes[key_group][shape_key]
    shape_key_slash = shape_key.replace('.', '/')
    path = path_dir + f'/module/{shape_key_slash}'
    tensor = repartition_tensor(source_shape, target_shape, path, target_mp_size, rank)
    ckpt['module']['language_model']['embedding']['word_embeddings']['weight'] = tensor

    # Set non-tensor, non-primitive values in ckpt
    search_ckpt(ckpt, path_dir.format(requested_rank=requested_rank))

    # random_rng_state must be a tuple
    ckpt['random_rng_state'][1] = tuple(ckpt['random_rng_state'][1])
    ckpt['random_rng_state'] = tuple(ckpt['random_rng_state'])

    # Set sizes to target
    ckpt['mp_world_size'] = target_mp_size
    ckpt['args'].world_size = target_mp_size
    ckpt['args'].model_parallel_size = target_mp_size

    # Set rank
    ckpt['args'].rank = rank
    ckpt['args'].local_rank = rank

    # Set target shapes
    ckpt['param_shapes'] = target_shapes

    return ckpt


def repartition_model_state(args):
    path = f"{args.shapes_dir}/shapes_mp{args.target_mp_size}.json"
    with open(path, "r") as json_file:
        target_shapes = json.load(json_file)

    path = f"{args.shapes_dir}/shapes_mp{args.source_mp_size}.json"
    with open(path, "r") as json_file:
        source_shapes = json.load(json_file)

    checkpoint = repartition_model(source_shapes,
                                   target_shapes,
                                   args)
    save_model_ckpt(checkpoint, args.output_dir, args.rank)


def repartition_flat_tensor(source_dim, target_dim, target_mp_size, rank, path,
                            source_flat, tensor_source_shape, dim,
                            source_offset, tensor_target_shape):
    reqs = map_requests(source_dim, target_dim, target_mp_size, rank)
    reqs = source_adjust_requests(reqs, source_dim)
    device_tensors = []

    for device, r in reqs.items():
        device_path = path.format(requested_rank=device)
        range_str = f'[{source_offset}:{source_offset + source_flat}]'
        device_tensor = get_tensor(device_path, range_str)
        device_tensor = device_tensor.reshape(tensor_source_shape)
        if dim == 0:
            device_tensor = device_tensor[r[0]:r[1]]
        elif dim == 1:
            device_tensor = device_tensor[:, r[0]:r[1]]
        else:
            raise NotImplementedError('Slice above second dimension')
        device_tensors.append(device_tensor)

    merged_tensor = torch.cat(device_tensors, dim=dim)

    assert tuple(merged_tensor.shape) == tuple(tensor_target_shape)

    flat_tensor = merged_tensor.flatten()

    return flat_tensor


def repartition_group_tensors(source_shapes, target_shapes, target_mp_size,
                              rank, path):
    source_offset = 0
    target_offset = 0
    group_tensors = []

    for tensor_key in target_shapes.keys():
        tensor_source_shape = source_shapes[tensor_key]
        tensor_target_shape = target_shapes[tensor_key]
        source_flat = int(np.prod(tensor_source_shape))
        if tensor_source_shape == tensor_target_shape:  # replicate
            requested_rank = 0  # TODO use random rank
            device_path = path.format(requested_rank=requested_rank)
            range_str = f'[{source_offset}:{source_offset + source_flat}]'
            device_tensor = get_tensor(device_path, range_str)
            assert (source_flat,) == tuple(device_tensor.shape)
            group_tensors.append(device_tensor)
        else:  # repartition
            for dim, (source_dim, target_dim) in enumerate(zip(tensor_source_shape,
                                                               tensor_target_shape)):
                if source_dim != target_dim:
                    flat_tensor = repartition_flat_tensor(source_dim,
                                                          target_dim,
                                                          target_mp_size,
                                                          rank,
                                                          path,
                                                          source_flat,
                                                          tensor_source_shape,
                                                          dim,
                                                          source_offset,
                                                          tensor_target_shape)

                    group_tensors.append(flat_tensor)
                    break  # assume only partitioned along 1 axis

        source_offset = source_offset + source_flat
        target_flat = int(np.prod(tensor_target_shape))
        target_offset = target_offset + target_flat

    return group_tensors


def repartition_zero_ckpt(source_shapes, target_shapes, args):
    """ Repartition Deepspeed Zero checkpoint """
    target_mp_size = args.target_mp_size
    rank = args.rank
    global_step = args.global_step
    requested_rank = random.randint(0, args.source_mp_size - 1)
    path_dir = f'global_step{global_step}'
    path_dir += '/zero_pp_rank_0_mp_rank_{requested_rank:02d}_optim_states.pt'
    ckpt = get_struct(path_dir.format(requested_rank=requested_rank))

    # base optimizer state: iterate through all param groups
    states = ckpt['optimizer_state_dict']['base_optimizer_state']['state']
    new_states = {}
    for i_str, state in states.items():
        i = int(i_str)
        repartitioned_state = {}

        for param in state.keys():
            if param == 'step':
                repartitioned_state[param] = state[param]
                continue

            path = path_dir + f'/optimizer_state_dict/base_optimizer_state/state/{i}/{param}'
            group_source_shapes = source_shapes[i]
            group_target_shapes = target_shapes[i]
            group_tensors = repartition_group_tensors(group_source_shapes,
                                                      group_target_shapes,
                                                      target_mp_size,
                                                      rank,
                                                      path)
            group_tensor = torch.cat(group_tensors)
            repartitioned_state[param] = group_tensor

        new_states[i] = repartitioned_state

    ckpt['optimizer_state_dict']['base_optimizer_state']['state'] = new_states

    # single_partition_of_fp32_groups
    fp32_groups = ckpt['optimizer_state_dict']['single_partition_of_fp32_groups']
    for i in range(len(fp32_groups)):
        path = path_dir + f'/optimizer_state_dict/single_partition_of_fp32_groups/{i}'
        group_tensors = repartition_group_tensors(source_shapes[i],
                                                  target_shapes[i],
                                                  target_mp_size,
                                                  rank,
                                                  path)
        tensor = torch.cat(group_tensors)
        fp32_groups[i] = tensor

    # Set non-tensor, non-primitive values in ckpt
    search_ckpt(ckpt, path_dir.format(requested_rank=requested_rank))

    return ckpt


def repartition_optimiser_state(args):
    with open(f"{args.shapes_dir}/shapes_mp{args.target_mp_size}.json",
              "r") as json_file:
        target_shapes = json.load(json_file)

    path = f"{args.shapes_dir}/shapes_mp{args.source_mp_size}.json"
    with open(path, "r") as json_file:
        source_shapes = json.load(json_file)

    checkpoint = repartition_zero_ckpt(source_shapes, target_shapes, args)

    save_optimiser_ckpt(checkpoint, args.output_dir, args.rank)


def repartition():
    parser = argparse.ArgumentParser(description='Repartition')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--shapes-dir', type=str)
    parser.add_argument('--source-mp-size', type=int)
    parser.add_argument('--target-mp-size', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--global-step', type=int)
    args = parser.parse_args()

    assert args.rank < args.target_mp_size

    # repartition
    print('repartition model')
    repartition_model_state(args)
    print('repartition optimiser')
    repartition_optimiser_state(args)


if __name__ == '__main__':
    repartition()
