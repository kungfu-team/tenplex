import argparse
import copy
import glob
import json
import os
import pickle
import re
import threading
from typing import Optional, Tuple, Union

import numpy as np
import requests
import torch


class Config():

    def __init__(self):
        self.input_dir = ''
        self.output_dir = ''
        self.ckpt_struct_dir = ''
        self.source_mp_degree = 0
        self.target_mp_degree = 0
        self.source_pp_degree = 0
        self.target_pp_degree = 0
        self.source_size = 0
        self.target_size = 0
        self.source_dp_degree = 0
        self.target_dp_degree = 0
        self.precision = ''


class CheckpointClient():

    def __init__(self, hosts: list, port: int, gpus_per_host: int,
                 config: Config):
        self.hosts = hosts
        self.port = port
        self.gpus_per_host = gpus_per_host
        self.config = config

    def _get_host_ip(self, rank: int) -> str:
        return self.hosts[rank // self.gpus_per_host]

    def get_tensor(self,
                   rank: int,
                   pp_rank: int,
                   mp_rank: int,
                   path: str,
                   rang: Optional[list] = None,
                   dim: Optional[int] = None) -> torch.Tensor:
        ip = self._get_host_ip(rank)
        url = f'http://{ip}:{self.port}'
        if self.config.source_pp_degree == 1:
            dir_name = f'mp_rank_{mp_rank:02d}'
        else:
            dir_name = f'mp_rank_{mp_rank:02d}_{pp_rank:03d}'
        url = os.path.join(url, dir_name)
        url = os.path.join(url, 'model_optim_rng.pt')
        url = os.path.join(url, path)
        if rang:
            if dim == 0:  # first dimension
                r_str = str(rang)
                r_str = r_str.replace(',', ':')
            elif dim == 1:  # second dimenson
                r_str = f'[:,{rang[0]}:{rang[1]}]'
            else:
                raise NotImplementedError(
                    'Requesting tensor along dimension higher 2')
            payload = {'range': r_str}
        else:
            payload = None
        result = requests.get(url, params=payload)
        tensor = pickle.loads(result.content)

        return tensor


def map_requests(source_dim, target_dim, source_mp_degree, target_mp_rank):
    """ returns the device
    and lower and upper bound for a merged model """
    lower_bound = target_mp_rank * target_dim
    upper_bound = target_mp_rank * target_dim + target_dim
    reqs = {}
    for device in range(source_mp_degree):
        device_lower_bound = device * source_dim
        device_upper_bound = (device + 1) * source_dim
        if lower_bound < device_lower_bound:  # before
            if upper_bound < device_lower_bound:  # before
                pass
            elif device_lower_bound < upper_bound <= device_upper_bound:
                # within
                reqs[device] = [device_lower_bound, upper_bound]
            elif device_upper_bound < upper_bound:  # after
                reqs[device] = [device_lower_bound, device_upper_bound]
        elif device_lower_bound <= lower_bound < device_upper_bound:  # within
            if upper_bound < device_lower_bound:  # before
                raise ValueError((f'upper_bound={upper_bound} is smaller ' +
                                  f'than lower_bound={lower_bound}'))
            if device_lower_bound < upper_bound <= device_upper_bound:
                # within
                reqs[device] = [lower_bound, upper_bound]
            elif device_upper_bound < upper_bound:  # after
                reqs[device] = [lower_bound, device_upper_bound]
        elif device_upper_bound < lower_bound:  # after
            pass

    return reqs


def source_adjust_requests(reqs, source_dim):
    """ Adjust the range to the source dimension """
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


def get_value(ckpt, keys):
    """ Get value within an checkpoint """
    value = ckpt
    for key in keys:
        if isinstance(key, int):
            try:
                value = value[key]
            except KeyError:
                value = value[str(key)]
        elif isinstance(key, str):
            try:
                value = value[key]
            except TypeError:
                value = value[int(key)]
            except KeyError:
                value = value[int(key)]
        else:
            raise ValueError(f'key {key} is not int nor str')

    return value


def get_range(tensor, rang, dim):
    """ Get the range of a tensor at a dimension """
    if rang[0] < 0:
        raise IndexError(f'lower bound {rang[0]} is small 0')
    if rang[1] > tensor.size(dim):
        raise IndexError((f'upper bound {rang[1]} is larger ' +
                          f'than tensor size {tensor.size(dim)}'))
    if dim == 0:
        return tensor[rang[0]:rang[1]]
    if dim == 1:
        return tensor[:, rang[0]:rang[1]]

    raise NotImplementedError('Dimension is larger than 1')


def create_dummy_tokens(num_missing_tokens, token_length):
    """ Create dummy tokens for given shape """
    if token_length == 0:
        padding = torch.ones((num_missing_tokens, ))
    else:
        padding = torch.ones((num_missing_tokens, token_length))

    return padding


class RankMap:

    def __init__(self, config: Config):
        shapes_dir = os.path.join(config.ckpt_struct_dir,
                                  f'pp{config.source_pp_degree:02d}')
        shapes_dir = os.path.join(shapes_dir,
                                  f'mp{config.source_mp_degree:02d}')
        shapes_dir = os.path.join(shapes_dir,
                                  f'dp{config.source_dp_degree:02d}')

        with open(os.path.join(shapes_dir, 'rank_map.json')) as json_file:
            json_mapping = json.load(json_file)

        self.rank_to_pp_mp_rank = dict()
        self.pp_mp_rank_to_rank = dict()
        for ke, val in json_mapping.items():
            rank = int(ke)
            pp_rank = val['pp_rank']
            mp_rank = val['mp_rank']
            self.rank_to_pp_mp_rank[rank] = (pp_rank, mp_rank)
            self.pp_mp_rank_to_rank[(pp_rank, mp_rank)] = rank

    def get_rank(self, pp_rank: int, mp_rank: int) -> int:
        """Look up mapping of (pp rank, mp rank) -> rank"""
        return self.pp_mp_rank_to_rank[(pp_rank, mp_rank)]

    def get_pp_mp_rank(self, rank: int) -> Tuple[int, int]:
        """Look up mapping of rank -> (pp rank, mp rank)"""
        return self.rank_to_pp_mp_rank[rank]


def replicate_tensor(source_pp_rank: int,
                     source_key_seq: list,
                     ckpt_client: CheckpointClient,
                     rank_map: RankMap,
                     source_mp_rank: int = 0) -> torch.Tensor:
    """Replicate the tensor of a checkpoint"""

    source_rank = rank_map.get_rank(source_pp_rank, source_mp_rank)
    source_key_seq_str = [str(k) for k in source_key_seq]
    path = '/'.join(source_key_seq_str)
    new_tensor = ckpt_client.get_tensor(source_rank, source_pp_rank,
                                        source_mp_rank, path)

    return new_tensor


def repartition_tensor(config: Config, source_shape: list, target_shape: list,
                       target_mp_rank: int, source_pp_rank: int,
                       source_key_seq: list, ckpt_client: CheckpointClient,
                       rank_map: RankMap) -> Optional[torch.Tensor]:
    """ Repartition a tensor of a layer """
    for dim, (source_dim,
              target_dim) in enumerate(zip(source_shape, target_shape)):
        if source_dim == target_dim:
            continue

        reqs = map_requests(source_dim, target_dim, config.source_mp_degree,
                            target_mp_rank)
        reqs = source_adjust_requests(reqs, source_dim)

        source_key_seq_str = [str(ele) for ele in source_key_seq]

        tensors = []
        for source_mp_rank, rang in reqs.items():
            source_rank = rank_map.get_rank(source_pp_rank, source_mp_rank)
            path = '/'.join(source_key_seq_str)
            source_tensor = ckpt_client.get_tensor(source_rank, source_pp_rank,
                                                   source_mp_rank, path, rang,
                                                   dim)
            tensors.append(source_tensor)
        new_tensor = torch.cat(tensors, dim=dim)

        return new_tensor

    return None


def get_layer_number(keys: Union[list, str]):
    if isinstance(keys, list):
        for key in keys:
            if isinstance(key, str):
                match = re.search(r'layers\.(\d+)\.', key)
                if match:
                    return int(match.group(1))
    elif isinstance(keys, str):
        match = re.search(r'layers\.(\d+)\.', keys)
        if match:
            return int(match.group(1))

    return None


def delete_key(target_state, key_seq):
    state = target_state
    for key in key_seq[:-1]:
        state = state[key]
    del state[key_seq[-1]]


def infer_source_pp_rank(config: Config, key_seq: Union[list, str]):
    if 'word_embeddings_for_head' in key_seq:
        return config.source_pp_degree - 1
    if 'embedding' in key_seq:
        return 0
    elif ('final_layernorm.weight' in key_seq
          or 'final_layernorm.bias' in key_seq or 'pooler' in key_seq
          or 'lm_head' in key_seq or 'binary_head' in key_seq):
        return config.source_pp_degree - 1
    else:
        return None


def layer_num_to_source_pp(config: Config, num_layers: int,
                           target_pp_rank: int, target_layer_number):
    target_num_layers = num_layers // config.target_pp_degree
    global_layer_number = target_layer_number + target_pp_rank * target_num_layers
    source_num_layers = num_layers // config.source_pp_degree
    source_pp_rank = None
    for rank in range(config.source_pp_degree):
        if rank * source_num_layers <= global_layer_number < (
                rank + 1) * source_num_layers:
            source_pp_rank = rank
            break
    if source_pp_rank is None:
        raise ValueError('source_pp_rank is None')

    return source_pp_rank


def construct_target_shapes_dir(config: Config):
    target_shapes_dir = os.path.join(config.ckpt_struct_dir,
                                     f'pp{config.target_pp_degree:02d}')
    target_shapes_dir = os.path.join(target_shapes_dir,
                                     f'mp{config.target_mp_degree:02d}')
    target_shapes_dir = os.path.join(target_shapes_dir,
                                     f'dp{config.target_dp_degree:02d}')

    return target_shapes_dir


def construct_source_shapes_dir(config: Config):
    source_shapes_dir = os.path.join(config.ckpt_struct_dir,
                                     f'pp{config.source_pp_degree:02d}')
    source_shapes_dir = os.path.join(source_shapes_dir,
                                     f'mp{config.source_mp_degree:02d}')
    source_shapes_dir = os.path.join(source_shapes_dir,
                                     f'dp{config.source_dp_degree:02d}')

    return source_shapes_dir


def get_shapes(shapes_dir: str, rank: int):
    rank_dir = os.path.join(shapes_dir, f'rank{rank:02d}/*')
    potential_files = glob.glob(rank_dir)
    if potential_files:
        shapes_file = potential_files[0]
    else:
        return None

    with open(shapes_file, 'r') as shape_fi:
        shapes = json.load(shape_fi)

    return shapes


def get_mpd_ranks(shapes_dir: str, rank: int, pp_degree: int):
    rank_dir = os.path.join(shapes_dir, f'rank{rank:02d}/*')
    potential_files = glob.glob(rank_dir)
    shapes_file = potential_files[0]
    file_name = os.path.basename(shapes_file)

    if pp_degree > 1:
        match = re.search(r'mp_rank_(\d+)_(\d+)', file_name)
        if match is None:
            raise ValueError('match is None')
        mp_rank = int(match.group(1))
        pp_rank = int(match.group(2))

        return pp_rank, mp_rank

    match = re.search(r'mp_rank_(\d+)', file_name)
    if match is None:
        raise ValueError('match is None')
    mp_rank = int(match.group(1))

    return None, mp_rank


def split_by_weight_decay(shapes: dict):
    weight_decay_tensors = []
    no_weight_decay_tensors = []

    name_keys = search_for_tensors(shapes['model'])
    if name_keys is None:
        raise ValueError('name_keys is None')
    names = ['.'.join(key) for key in name_keys]

    for name, key in zip(names, name_keys):
        full_key = ['model']
        full_key.extend(key)
        tensor_shape = get_value(shapes, full_key)['tensor']
        if 'bias' in name or len(tensor_shape) == 1:
            no_weight_decay_tensors.append(name)
        else:
            weight_decay_tensors.append(name)

    return weight_decay_tensors, no_weight_decay_tensors


def get_group_sizes(config: Config, shapes_dir: str, size: int,
                    pp_degree: int):
    group_sizes = dict()

    mp0_shape = None
    for pp_rank in range(pp_degree):
        for rank in range(size):
            if pp_degree == 1:
                tail = os.path.join(f'rank{rank:02d}', 'mp_rank_00.json')
            else:
                tail = os.path.join(f'rank{rank:02d}',
                                    f'mp_rank_00_{pp_rank:03d}.json')
            shape_path = os.path.join(shapes_dir, tail)
            if not os.path.exists(shape_path):
                continue

            with open(shape_path, 'r') as shapes_fi:
                mp0_shape = json.load(shapes_fi)
            break

        if mp0_shape is None:
            raise ValueError('mp0_shape is None')

        if config.precision == "fp16":
            group_sizes[pp_rank] = {
                0:
                len(mp0_shape['optimizer']['optimizer']['param_groups'][0]
                    ['params']),
                1:
                len(mp0_shape['optimizer']['optimizer']['param_groups'][1]
                    ['params'])
            }
        elif config.precision == "fp32":
            group_sizes[pp_rank] = {
                0: len(mp0_shape['optimizer']['param_groups'][0]['params']),
                1: len(mp0_shape['optimizer']['param_groups'][1]['params'])
            }

    return group_sizes


def is_layers_in(keys: list):
    for key in keys:
        if 'layers' in key:
            return True

    return False


def get_source_mp0_shapes(config: Config, source_shapes_dir: str,
                          source_pp_rank: int):
    for source_rank in range(config.source_size):
        if config.source_pp_degree == 1:
            tail = os.path.join(f'rank{source_rank:02d}', 'mp_rank_00.json')
        else:
            tail = os.path.join(f'rank{source_rank:02d}',
                                f'mp_rank_00_{source_pp_rank:03d}.json')
        shape_path = os.path.join(source_shapes_dir, tail)
        if not os.path.exists(shape_path):
            continue

        with open(shape_path, 'r') as shapes_fi:
            source_mp0_shape = json.load(shapes_fi)

        return source_mp0_shape

    return None


def is_rng_in_keys(keys):
    """ Check for RNG inside the key sequence """
    for key in keys:
        if isinstance(key, str):
            if 'rng' in key:
                return True
    return False


def set_tensor(ckpt, keys, tensor):
    """ Set the tensor inside a checkpoint """
    value = ckpt
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = tensor


def vocab_size_with_padding(orig_vocab_size,
                            tensor_model_parallel_size,
                            make_vocab_size_divisible_by=128):
    """Pad vocab size so it is divisible by model parallel size and
      still having GPU friendly size."""

    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1

    return after


def search_for_tensors(value, keys: list = []):
    """ Search for tensors within a checkpoint
    and return the key sequences to those tensors,
    except for RNG tensors """
    if isinstance(value, dict):
        new_tensor_keys = []
        for key, val in value.items():
            if key == 'tensor':
                if not is_rng_in_keys(keys):
                    return [keys]
                return None

            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            tensor_keys = search_for_tensors(val, new_keys)
            if tensor_keys:
                new_tensor_keys.extend(tensor_keys)
        return new_tensor_keys
    if isinstance(value, (list, set, tuple)):
        new_tensor_keys = []
        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(i)
            tensor_keys = search_for_tensors(val, new_keys)
            if tensor_keys:
                new_tensor_keys.extend(tensor_keys)
        return new_tensor_keys

    return None


def search_for_not_primitive_Nor_tensor(value, keys: list = []):
    """ Search for NotPrimitiveNorTensor within a checkpoint """
    if isinstance(value, str):
        if value == 'NotPrimitiveNorTensor':
            return [keys]
    if isinstance(value, dict):
        new_tensor_keys = []
        for key, val in value.items():
            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            tensor_keys = search_for_not_primitive_Nor_tensor(val, new_keys)
            if tensor_keys:
                new_tensor_keys.extend(tensor_keys)
        return new_tensor_keys
    if isinstance(value, (list, set, tuple)):
        new_tensor_keys = []
        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(i)
            tensor_keys = search_for_not_primitive_Nor_tensor(val, new_keys)
            if tensor_keys:
                new_tensor_keys.extend(tensor_keys)
        return new_tensor_keys

    return None


def transform_state_for_rank(target_rank: int, config: Config, num_layers: int,
                             vocab_size: int, hosts: list):
    """ Transform each layer of the model """

    print(f'Start transformation for target rank {target_rank}')

    target_shapes_dir = construct_target_shapes_dir(config)
    source_shapes_dir = construct_source_shapes_dir(config)

    target_group_sizes = get_group_sizes(config, target_shapes_dir,
                                         config.target_size,
                                         config.target_pp_degree)
    source_group_sizes = get_group_sizes(config, source_shapes_dir,
                                         config.source_size,
                                         config.source_pp_degree)

    target_shapes = get_shapes(target_shapes_dir, target_rank)
    if target_shapes is None:
        return

    ckpt_client = CheckpointClient(hosts, 21234, 4, config)
    rank_map = RankMap(config)

    target_pp_rank, target_mp_rank = get_mpd_ranks(target_shapes_dir,
                                                   target_rank,
                                                   config.target_pp_degree)
    if config.target_pp_degree == 1 and target_pp_rank is None:
        target_pp_rank = 0

    target_state = copy.deepcopy(target_shapes)

    tensor_key_seqs = search_for_tensors(target_state)

    if tensor_key_seqs is None:
        raise ValueError('tensor_key_seqs is None')

    weight_decay_tensors, no_weight_decay_tensors = split_by_weight_decay(
        target_shapes)

    for tensor_key_seq in tensor_key_seqs:
        #  print(f'rank {target_rank}: transform {tensor_key_seq}')

        key_seq_plus_tensor = copy.deepcopy(tensor_key_seq)
        key_seq_plus_tensor.append('tensor')
        target_shape = get_value(target_shapes, key_seq_plus_tensor)

        source_key_seq = None

        # if scalar just use pp_rank 0
        if len(target_shape) == 1 and target_shape[0] == 1:
            source_pp_rank = 0
        elif 'model' in tensor_key_seq:
            if is_layers_in(tensor_key_seq):
                target_layer_number = get_layer_number(tensor_key_seq)
                source_pp_rank = layer_num_to_source_pp(
                    config, num_layers, target_pp_rank, target_layer_number)

                source_key_seq = copy.deepcopy(tensor_key_seq)
                for i, key in enumerate(source_key_seq):
                    if 'layers' in key:
                        target_num_layers = num_layers // config.target_pp_degree
                        global_layer_number = target_layer_number + target_pp_rank * target_num_layers
                        source_num_layers = num_layers // config.source_pp_degree
                        source_layer_number = global_layer_number - source_pp_rank * source_num_layers
                        source_key_seq[i] = re.sub(
                            r'layers\.(\d)+', f'layers.{source_layer_number}',
                            key)
                        break
                source_key_seq.append('tensor')
            else:
                source_pp_rank = infer_source_pp_rank(config, tensor_key_seq)
        elif 'optimizer' in tensor_key_seq:
            if 'state' in tensor_key_seq:
                target_idx = None
                for key in tensor_key_seq:
                    try:
                        target_idx = int(key)
                    except ValueError:
                        continue
                    break

                target_group0_size = target_group_sizes[target_pp_rank][0]
                if target_idx < target_group0_size:
                    group_idx = target_idx
                    tensor_name = weight_decay_tensors[group_idx]
                    group = 0
                else:
                    group_idx = target_idx - target_group0_size
                    tensor_name = no_weight_decay_tensors[group_idx]
                    group = 1

                if 'layers' in tensor_name:
                    target_layer_number = get_layer_number(tensor_name)
                    source_pp_rank = layer_num_to_source_pp(
                        config, num_layers, target_pp_rank,
                        target_layer_number)
                else:
                    source_pp_rank = infer_source_pp_rank(config, tensor_name)

                global_idx = group_idx
                for prev_rank in range(target_pp_rank):
                    global_idx = global_idx + target_group_sizes[prev_rank][
                        group]
                prev_total_size = 0
                for prev_rank in range(source_pp_rank):
                    prev_total_size = prev_total_size + source_group_sizes[
                        prev_rank][group]
                source_idx = global_idx - prev_total_size

                if group == 1:
                    source_group0_size = source_group_sizes[source_pp_rank][0]
                    source_idx = source_idx + source_group0_size

                source_key_seq = copy.deepcopy(tensor_key_seq)
                for i, key in enumerate(source_key_seq):
                    if str(target_idx) == key:
                        source_key_seq[i] = str(source_idx)
                source_key_seq.append('tensor')

            else:
                group = int(tensor_key_seq[-2])
                target_idx = int(tensor_key_seq[-1])
                if group == 0:
                    tensor_name = weight_decay_tensors[target_idx]
                elif group == 1:
                    tensor_name = no_weight_decay_tensors[target_idx]
                else:
                    raise ValueError('group larger 1')

                if 'layers' in tensor_name:
                    target_layer_number = get_layer_number(tensor_name)
                    source_pp_rank = layer_num_to_source_pp(
                        config, num_layers, target_pp_rank,
                        target_layer_number)
                else:
                    source_pp_rank = infer_source_pp_rank(config, tensor_name)

                # get source_idx
                # group is equal
                global_idx = target_idx
                for prev_rank in range(target_pp_rank):
                    global_idx = global_idx + target_group_sizes[prev_rank][
                        group]
                prev_total_size = 0
                for prev_rank in range(source_pp_rank):
                    prev_total_size = prev_total_size + source_group_sizes[
                        prev_rank][group]
                source_idx = global_idx - prev_total_size

                source_key_seq = copy.deepcopy(tensor_key_seq)
                source_key_seq[-1] = str(source_idx)
                source_key_seq.append('tensor')
        else:
            raise NotImplementedError

        if source_pp_rank is None:
            raise ValueError('source_pp_rank is None')

        # load shapes of source mp rank 0 and source_layer_pp_rank
        source_mp0_shapes = get_source_mp0_shapes(config, source_shapes_dir,
                                                  source_pp_rank)

        if source_mp0_shapes is None:
            raise ValueError('source_mp0_shapes is None')

        # get shape of tensor_key_seq
        key_seq_plus_tensor = copy.deepcopy(tensor_key_seq)
        key_seq_plus_tensor.append('tensor')
        target_shape = get_value(target_shapes, key_seq_plus_tensor)

        if source_key_seq is None:
            source_key_seq = copy.deepcopy(key_seq_plus_tensor)
        source_shape = get_value(source_mp0_shapes, source_key_seq)

        if source_shape == target_shape:
            new_tensor = replicate_tensor(source_pp_rank, source_key_seq[:-1],
                                          ckpt_client, rank_map)
        else:
            new_tensor = repartition_tensor(config, source_shape, target_shape,
                                            target_mp_rank, source_pp_rank,
                                            source_key_seq[:-1], ckpt_client,
                                            rank_map)
            if new_tensor is None:
                raise ValueError('new_tensor is None')

            #  if ('word_embeddings' in source_key_seq or
            #      ('lm_head' in source_key_seq and 'bias' in source_key_seq)):
            new_tensor_shape = list(new_tensor.size())
            if new_tensor_shape != target_shape and new_tensor_shape[0] > 5000:
                print(f'{source_key_seq}: Assume padding for embedding needed')
                padded_length = vocab_size_with_padding(
                    vocab_size, config.target_mp_degree)
                if new_tensor_shape[0] != padded_length:
                    # add padding
                    num_paddings = target_shape[0] - new_tensor_shape[0]
                    if len(new_tensor_shape) == 1:
                        padding = create_dummy_tokens(num_paddings, 0)
                    else:
                        padding = create_dummy_tokens(num_paddings,
                                                      new_tensor_shape[1])
                    new_tensor = torch.cat([new_tensor, padding])

        if list(new_tensor.size()) != target_shape:
            raise ValueError(f'{source_key_seq} '
                             f'new tensor {new_tensor.size()} '
                             f'!= target shape {target_shape}')

        delete_key(target_state, key_seq_plus_tensor)
        set_tensor(target_state, tensor_key_seq, new_tensor)

    keys = search_for_not_primitive_Nor_tensor(target_state)

    if keys is None:
        raise ValueError('keys is None')

    source_rank = 0
    source_pp_rank = 0
    source_mp_rank = 0
    for key in keys:
        key_str = [str(k) for k in key]
        path = '/'.join(key_str)
        value = ckpt_client.get_tensor(source_rank, source_pp_rank,
                                       source_mp_rank, path)
        #  value = get_value(source_ckpt, key)
        set_tensor(target_state, key, value)

    iteration = ckpt_client.get_tensor(source_rank, source_pp_rank,
                                       source_mp_rank, 'iteration')
    target_state['iteration'] = iteration

    # set arguments
    target_state['args'].iteration = iteration
    target_state['args'].padded_vocab_size = vocab_size_with_padding(
        vocab_size, config.target_mp_degree)
    target_state['args'].tensor_model_parallel_size = config.target_mp_degree
    target_state['args'].pipeline_model_parallel_size = config.target_pp_degree
    target_state[
        'args'].transformer_pipeline_model_parallel_size = config.target_pp_degree
    target_state['args'].rank = target_rank

    # set RNG state
    target_state['rng_state'] = ckpt_client.get_tensor(source_rank,
                                                       source_pp_rank,
                                                       source_mp_rank,
                                                       'rng_state')

    # set optimiser parameter schduler
    target_state['opt_param_scheduler'] = ckpt_client.get_tensor(
        source_rank, source_pp_rank, source_mp_rank, 'opt_param_scheduler')
    if config.precision == "fp16":
        length = len(target_state['optimizer']['optimizer']['param_groups'])
    elif config.precision == "fp32":
        length = len(target_state['optimizer']['param_groups'])
    else:
        raise NotImplementedError(
            f"precision {config.precision} is not implemented")
    # Set optimiser param_groups
    for group in range(length):
        if config.precision == "fp16":
            keys = ['optimizer', 'optimizer', 'param_groups', group]
        elif config.precision == "fp32":
            keys = ['optimizer', 'param_groups', group]
        else:
            raise NotImplementedError(
                f"precision {config.precision} is not implemented")
        group_params = get_value(target_state, keys)
        for attr in group_params.keys():
            if attr == "params":
                # skip params
                continue
            group_attr_keys = copy.deepcopy(keys)
            group_attr_keys.append(attr)
            keys_str = [str(k) for k in group_attr_keys]
            path = '/'.join(keys_str)
            value = ckpt_client.get_tensor(source_rank, source_pp_rank,
                                           source_mp_rank, path)
            set_tensor(target_state, group_attr_keys, value)
    # Set optimiser grad_scaler
    if config.precision == "fp16":
        target_state['optimizer']['grad_scaler'] = ckpt_client.get_tensor(
            source_rank, source_pp_rank, source_mp_rank,
            'optimizer/grad_scaler')

    if config.target_pp_degree == 1:
        tail = f'mp_rank_{target_mp_rank:02d}'
    else:
        tail = f'mp_rank_{target_mp_rank:02d}_{target_pp_rank:03d}'
    target_state_base_path = os.path.join(config.output_dir, tail)
    os.makedirs(target_state_base_path)
    target_state_path = os.path.join(target_state_base_path,
                                     'model_optim_rng.pt')
    torch.save(target_state, target_state_path)


def transform_state(config: Config,
                    num_layers=24,
                    vocab_size=30524,
                    use_threads=False):
    if use_threads:
        threads = []
        for target_rank in range(config.target_size):
            t = threading.Thread(target=transform_state_for_rank,
                                 name=f'thread-{target_rank:02d}',
                                 args=(target_rank, config, num_layers,
                                       vocab_size))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    else:
        for target_rank in range(config.target_size):
            transform_state_for_rank(target_rank, config, num_layers,
                                     vocab_size)


def main():
    """ Parse arguments and start transformation """
    parser = argparse.ArgumentParser(description='Transformation')
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--ckpt-struct-dir', type=str)
    parser.add_argument('--source-mp-degree', type=int)
    parser.add_argument('--target-mp-degree', type=int)
    parser.add_argument('--source-pp-degree', type=int)
    parser.add_argument('--target-pp-degree', type=int)
    parser.add_argument('--source-size', type=int)
    parser.add_argument('--target-size', type=int)
    parser.add_argument('--target-rank', type=int)
    parser.add_argument('--precision', type=str)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--hosts', type=str)
    args = parser.parse_args()

    config = Config()
    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    config.ckpt_struct_dir = args.ckpt_struct_dir
    config.source_mp_degree = args.source_mp_degree
    config.target_mp_degree = args.target_mp_degree
    config.source_pp_degree = args.source_pp_degree
    config.target_pp_degree = args.target_pp_degree
    config.source_size = args.source_size
    config.target_size = args.target_size
    source_pp_mp_degree = config.source_pp_degree * config.source_mp_degree
    config.source_dp_degree = config.source_size // source_pp_mp_degree
    target_pp_mp_degree = config.target_pp_degree * config.target_mp_degree
    config.target_dp_degree = config.target_size // target_pp_mp_degree
    config.precision = args.precision

    num_layers = args.num_layers
    vocab_size = 30524
    #  hosts = ['10.10.10.1', '10.10.10.3', '10.10.10.4']
    hosts = args.hosts.split(',')
    transform_state_for_rank(args.target_rank, config, num_layers, vocab_size,
                             hosts)


if __name__ == '__main__':
    main()
