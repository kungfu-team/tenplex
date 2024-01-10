import argparse
import copy
import glob
import json
import os
import pickle
import re
from typing import Optional

import numpy as np
import requests
import torch


class Config:

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


class CheckpointClient():

    def __init__(
        self,
        config: Config,
        hosts: list,
        port: int = 21234,
        gpus_per_host: int = 4,
    ):
        self.hosts = hosts
        self.port = port
        self.gpus_per_host = gpus_per_host
        self.config = config

    def _get_host_ip(self, rank: int) -> str:
        return self.hosts[rank // self.gpus_per_host]

    def get_tensor(self,
                   rank: int,
                   path: str,
                   rang: Optional[list] = None,
                   dim: Optional[int] = None) -> torch.Tensor:
        ip = self._get_host_ip(rank)
        url = f'http://{ip}:{self.port}'
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

    def get_struct(self, rank, path):
        ip = self._get_host_ip(rank)
        url = f'http://{ip}:{self.port}'
        url = os.path.join(url, path)
        result = requests.get(url)
        struct = json.loads(result.content)

        return struct

    def get_ckpt(self, rank, path):
        ip = self._get_host_ip(rank)
        url = f'http://{ip}:{self.port}'
        url = os.path.join(url, path)
        result = requests.get(url)
        return pickle.loads(result.content)


class LayerMap:

    def __init__(self, config: Config):
        shapes_dir = os.path.join(config.ckpt_struct_dir,
                                  f'pp{config.source_pp_degree:02d}')
        shapes_dir = os.path.join(shapes_dir,
                                  f'mp{config.source_mp_degree:02d}')
        shapes_dir = os.path.join(shapes_dir,
                                  f'dp{config.source_dp_degree:02d}')

        with open(os.path.join(shapes_dir, 'layer_map.json')) as json_file:
            json_mapping = json.load(json_file)

        self.rank_to_tp_rank_layers = dict()
        self.tp_rank_layer_to_rank = dict()
        for ke, val in json_mapping.items():
            rank = int(ke)
            tp_rank = val['tp_rank']
            layers = val['layer_numbers']
            self.rank_to_tp_rank_layers[rank] = (tp_rank, layers)
            for layer in layers:
                self.tp_rank_layer_to_rank[(tp_rank, layer)] = rank

    def get_rank(self, tp_rank: int, layer: int) -> int:
        """Look up mapping of (pp rank, mp rank) -> rank"""
        return self.tp_rank_layer_to_rank[(tp_rank, layer)]

    def get_tp_rank_layers(self, rank: int):
        """Look up mapping of rank -> (pp rank, mp rank)"""
        return self.rank_to_tp_rank_layers[rank]


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
        try:
            value = value[key]
        except KeyError:
            value = value[str(key)]

    return value


def get_offset(shapes, module, param):
    """ Calcluate the offset of the source tensor """
    offset = 0
    for module_i, module_shapes in shapes.items():
        for param_i, shape in module_shapes.items():
            if module_i == module and param_i == param:
                return offset
            offset = offset + np.prod(shape)
    return None


def get_source_attr(shapes, group, module, param):
    """ Get the attributes of all source ranks """
    attr = []
    for rank, _ in shapes.items():
        try:
            shape = shapes[rank][group][module][param]
        except KeyError:
            continue
        rank_offset = get_offset(shapes[rank][group], module, param)
        attr.append({'rank': rank, 'offset': rank_offset, 'shape': shape})

    return attr


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
    padding = torch.zeros((num_missing_tokens, token_length))

    return padding


def replicate_layer_tensor(ckpt_client: CheckpointClient,
                           layer_map: LayerMap,
                           layer_number: int,
                           target_shape_name: str,
                           source_tp_rank: int = 0):
    """ Replicate the tensor of a layer"""

    source_layer_name = (f'layer_{layer_number:02d}-' +
                         f'model_{source_tp_rank:02d}-model_states.pt')
    path = os.path.join(source_layer_name, target_shape_name)
    rank = layer_map.get_rank(source_tp_rank, layer_number)
    new_tensor = ckpt_client.get_tensor(rank, path)

    return new_tensor


def repartition_layer_tensor(config: Config, ckpt_client: CheckpointClient,
                             layer_map: LayerMap, source_shape: list,
                             target_shape: list, target_mp_rank: int,
                             target_shape_name: str, layer_number: int):
    """ Repartition a tensor of a layer """
    for dim, (source_dim,
              target_dim) in enumerate(zip(source_shape, target_shape)):
        if source_dim == target_dim:
            continue

        reqs = map_requests(source_dim, target_dim, config.source_mp_degree,
                            target_mp_rank)
        reqs = source_adjust_requests(reqs, source_dim)

        tensors = []
        for source_mp_rank, rang in reqs.items():
            source_layer_name = (f'layer_{layer_number:02d}-' +
                                 f'model_{source_mp_rank:02d}-model_states.pt')
            path = os.path.join(source_layer_name, target_shape_name)
            rank = layer_map.get_rank(source_mp_rank, layer_number)
            source_tensor = ckpt_client.get_tensor(rank, path, rang, dim)
            tensors.append(source_tensor)
        new_tensor = torch.cat(tensors, dim=dim)

        size_list = list(new_tensor.size())
        if ('word_embeddings' in target_shape_name
                and size_list[dim] < target_shape[dim]):
            num_missing_tokens = target_shape[dim] - size_list[dim]
            padding = create_dummy_tokens(num_missing_tokens, target_shape[1])
            new_tensor = torch.cat([new_tensor, padding])
            size_list = list(new_tensor.size())
        if size_list != target_shape:
            error_str = (f'layer {layer_number}, {target_shape_name}' +
                         f' is {size_list} != {target_shape} should')
            raise ValueError(error_str)

        return new_tensor


def transform_layer(config: Config, ckpt_client: CheckpointClient,
                    layer_map: LayerMap, source_shapes: dict,
                    target_shapes: dict, layer_number: int,
                    target_mp_rank: int):
    """ Transform a layer """
    target_layer = {}
    for target_shape_name, target_shape in target_shapes.items():
        source_shape = source_shapes[target_shape_name]

        if source_shape == target_shape:
            new_tensor = replicate_layer_tensor(ckpt_client, layer_map,
                                                layer_number,
                                                target_shape_name)
        else:
            new_tensor = repartition_layer_tensor(config, ckpt_client,
                                                  layer_map, source_shape,
                                                  target_shape, target_mp_rank,
                                                  target_shape_name,
                                                  layer_number)
        target_layer[target_shape_name] = new_tensor

    return target_layer


def transform_layers(config: Config, target_rank: int):
    """ Transform each layer of the model """
    target_shapes_dir = os.path.join(config.ckpt_struct_dir,
                                     f'pp{config.target_pp_degree:02d}')
    target_shapes_dir = os.path.join(target_shapes_dir,
                                     f'mp{config.target_mp_degree:02d}')
    target_shapes_dir = os.path.join(target_shapes_dir,
                                     f'dp{config.target_dp_degree:02d}')

    source_shapes_dir = os.path.join(config.ckpt_struct_dir,
                                     f'pp{config.source_pp_degree:02d}')
    source_shapes_dir = os.path.join(source_shapes_dir,
                                     f'mp{config.source_mp_degree:02d}')
    source_shapes_dir = os.path.join(source_shapes_dir,
                                     f'dp{config.source_dp_degree:02d}')

    print(f'transform layers for rank {target_rank}')

    # FIX: make hosts dynamic
    hosts = ['10.10.10.3', '10.10.10.1', '10.10.10.4']
    ckpt_client = CheckpointClient(config, hosts)
    layer_map = LayerMap(config)

    target_rank_shapes_dir = os.path.join(target_shapes_dir,
                                          f'rank{target_rank:02d}')

    target_layer_files = glob.glob(target_rank_shapes_dir + '/layer_*')
    target_layer_files.sort()

    for target_layer_file in target_layer_files:
        layer_basename = os.path.basename(target_layer_file)
        match = re.match(r'layer_(\d+)-model_(\d+)-model_states',
                         layer_basename)
        if match is None:
            raise ValueError('match is None')
        layer_number = int(match.group(1))
        target_mp_rank = int(match.group(2))

        with open(target_layer_file, 'r') as target_layer_fi:
            target_shapes = json.load(target_layer_fi)

        source_layer_files = None
        for source_rank in range(config.source_size):
            source_shapes_rank_dir = os.path.join(source_shapes_dir,
                                                  f'rank{source_rank:02d}')
            glob_path = os.path.join(source_shapes_rank_dir,
                                     f'layer_{layer_number:02d}*')
            source_layer_files = glob.glob(glob_path)
            if len(source_layer_files) > 0:
                break
        if source_layer_files is None:
            raise ValueError('source_layer_files is None')
        if not source_layer_files:
            raise ValueError('source_layer_files is empty')
        source_layer_file = source_layer_files[0]
        with open(source_layer_file, 'r') as source_layer_fi:
            source_shapes = json.load(source_layer_fi)

        target_layer = transform_layer(config, ckpt_client, layer_map,
                                       source_shapes, target_shapes,
                                       layer_number, target_mp_rank)

        target_layer_name = (f'layer_{layer_number:02d}-' +
                             f'model_{target_mp_rank:02d}-model_states.pt')
        target_layer_path = os.path.join(config.output_dir, target_layer_name)
        torch.save(target_layer, target_layer_path)


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


def search_for_tensors(value, keys=None):
    """ Search for tensors within a checkpoint
    and return the key sequences to those tensors,
    except for RNG tensors """
    if keys is None:
        keys = []
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


def search_for_not_primitive_Nor_tensor(value, keys=None):
    """ Search for NotPrimitiveNorTensor within a checkpoint """
    if keys is None:
        keys = []
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


def get_target_size(shapes: dict):
    """ Get the total size of all shapes within a optimiser parameter group """
    size = 0
    for _, module_shapes in shapes.items():
        for _, param_shape in module_shapes.items():
            size = size + np.prod(param_shape)

    return size


def get_source_shapes(config, source_shapes_dir):
    """ Get the source optimiser parameter shapes of all source ranks """
    source_shapes = {}

    for source_rank in range(config.source_size):
        tail = os.path.join(f'rank{source_rank:02d}',
                            f'optimiser_param_shapes_{source_rank:02d}.json')
        source_rank_path = os.path.join(source_shapes_dir, tail)
        with open(source_rank_path, 'r') as shapes_fi:
            source_rank_shapes = json.load(shapes_fi)
        source_shapes[source_rank] = source_rank_shapes

    return source_shapes


def get_target_mp_rank(config, rank):
    """ Get the target mp rank within a pp group """
    pp_group_size = config.target_mp_degree * config.target_dp_degree
    pp_group = None
    for group in range(config.target_pp_degree):
        if group * pp_group_size <= rank < (group + 1) * pp_group_size:
            pp_group = group
            break
    if pp_group is None:
        raise ValueError('Shard is None')
    pp_group_rank = rank % pp_group_size
    if pp_group_rank >= config.target_mp_degree:
        return None
    return pp_group_rank + pp_group * config.target_mp_degree


def get_rank0_struct(config: Config,
                     ckpt_client: CheckpointClient,
                     file_name: str = 'mp_rank_00_model_states.pt'):
    """ Get the checkpoint of source rank 0 """
    ckpt = ckpt_client.get_struct(0, file_name)
    return ckpt


def get_target_rank_shapes(target_shapes_dir: str, rank: int) -> dict:
    """ Read the optimiser parameter shapes """
    tail = os.path.join(f'rank{rank:02d}',
                        f'optimiser_param_shapes_{rank:02d}.json')
    target_rank_shapes_path = os.path.join(target_shapes_dir, tail)
    with open(target_rank_shapes_path, 'r') as shapes_fi:
        target_rank_shapes = json.load(shapes_fi)

    return target_rank_shapes


def get_group_key(key_seq: list):
    """ Get the key according to the optimiser parameter group number """
    if 0 in key_seq:
        group_key = 'weight_decay'
    elif 1 in key_seq:
        group_key = 'no_weight_decay'
    else:
        raise ValueError(f'no number in keys {key_seq}')

    return group_key


def repartition_flat_tensor(config: Config,
                            ckpt_client: CheckpointClient,
                            tensor_source_shape: list,
                            tensor_target_shape: list,
                            rank: int,
                            keys: list,
                            source_attrs: list,
                            is_embedding: bool = False):
    """ Repartition a flat tensor for the optimiser state """
    source_layer_ranks = [attr['rank'] for attr in source_attrs]
    source_flat = np.prod(tensor_source_shape)
    source_mp_degree = config.source_mp_degree
    source_offset = source_attrs[0]['offset']
    target_mp_rank = rank % config.target_mp_degree

    for dim, (source_dim, target_dim) in enumerate(
            zip(tensor_source_shape, tensor_target_shape)):
        if source_dim != target_dim:
            reqs = map_requests(source_dim, target_dim, source_mp_degree,
                                target_mp_rank)
            reqs = source_adjust_requests(reqs, source_dim)

            device_tensors = []
            for source_mp_rank, rang in reqs.items():
                source_rank = source_layer_ranks[source_mp_rank]
                file_name = f'mp_rank_{source_rank:02d}_model_states.pt'
                keys_str = [str(k) for k in keys]
                key_path = '/'.join(keys_str)
                path = os.path.join(file_name, key_path)
                source_range = [source_offset, source_offset + source_flat]
                device_tensor = ckpt_client.get_tensor(source_rank, path,
                                                       source_range, 0)
                device_tensor = device_tensor.reshape(tensor_source_shape)
                device_tensor = get_range(device_tensor, rang, dim)
                device_tensors.append(device_tensor)

            merged_tensor = torch.cat(device_tensors, dim=dim)

            size_list = list(merged_tensor.size())
            if (is_embedding and size_list[dim] < tensor_target_shape[dim]):
                # Add dummy tokens if there are not enough
                num_missing_tokens = tensor_target_shape[dim] - size_list[dim]
                padding = create_dummy_tokens(num_missing_tokens,
                                              tensor_target_shape[1])
                merged_tensor = torch.cat([merged_tensor, padding])
                size_list = list(merged_tensor.size())

            if size_list != tensor_target_shape:
                error_str = (f'merged tensor {size_list} ' +
                             f'!= target {tensor_target_shape}')
                raise ValueError(error_str)

            flat_tensor = merged_tensor.flatten()

            return flat_tensor


def replicate_flat_tensor(config: Config, ckpt_client: CheckpointClient,
                          source_attr: dict, keys: list):
    """ Replicate a flat tensor for the optimiser state """
    source_flat = np.prod(source_attr['shape'])
    source_offset = source_attr['offset']
    source_rank = source_attr['rank']

    source_tp_rank = source_rank // config.source_dp_degree  # TODO: verify if calc correct
    file_name = f'mp_rank_{source_tp_rank:02d}_model_states.pt'
    keys_str = [str(k) for k in keys]
    key_path = '/'.join(keys_str)
    source_ckpt_path = os.path.join(file_name, key_path)
    lower_bound = source_offset
    upper_bound = source_offset + source_flat
    source_range = [lower_bound, upper_bound]
    device_tensor = ckpt_client.get_tensor(source_rank, source_ckpt_path,
                                           source_range, 0)

    return device_tensor


def repartition_optimiser_group_tensor(config: Config,
                                       ckpt_client: CheckpointClient,
                                       source_shapes: dict,
                                       target_shapes: dict, rank: int,
                                       keys: list, group_key: str):
    """ Repartition a tensor of a optimiser group """
    target_group_shapes = target_shapes[group_key]
    group_tensors = []

    for module_name, module in target_group_shapes.items():
        for param_name, _ in module.items():
            tensor_target_shape = target_group_shapes[module_name][param_name]
            source_attrs = get_source_attr(source_shapes, group_key,
                                           module_name, param_name)
            source_attr = source_attrs[0]
            tensor_source_shape = source_attr['shape']
            source_flat = np.prod(source_attr['shape'])

            if tensor_source_shape == tensor_target_shape:
                device_tensor = replicate_flat_tensor(config, ckpt_client,
                                                      source_attr, keys)

                if (source_flat, ) != tuple(device_tensor.size()):
                    raise ValueError(f'{module_name} {param_name}, '
                                     f'source_flat {source_flat} != '
                                     f'tensor shape {device_tensor.size(0)}')

                group_tensors.append(device_tensor)
            else:
                is_embedding = 'word_embeddings' in module_name
                flat_tensor = repartition_flat_tensor(config, ckpt_client,
                                                      tensor_source_shape,
                                                      tensor_target_shape,
                                                      rank, keys, source_attrs,
                                                      is_embedding)

                group_tensors.append(flat_tensor)
                break  # assume only partitioned along 1 axis

    return torch.cat(group_tensors)


def transform_state(config: Config, target_rank: int):
    """ Transform the optimiser state """
    target_shapes_dir = os.path.join(config.ckpt_struct_dir,
                                     f'pp{config.target_pp_degree:02d}')
    target_shapes_dir = os.path.join(target_shapes_dir,
                                     f'mp{config.target_mp_degree:02d}')
    target_shapes_dir = os.path.join(target_shapes_dir,
                                     f'dp{config.target_dp_degree:02d}')

    source_shapes_dir = os.path.join(config.ckpt_struct_dir,
                                     f'pp{config.source_pp_degree:02d}')
    source_shapes_dir = os.path.join(source_shapes_dir,
                                     f'mp{config.source_mp_degree:02d}')
    source_shapes_dir = os.path.join(source_shapes_dir,
                                     f'dp{config.source_dp_degree:02d}')

    # FIX: make hosts dynamic
    hosts = ['10.10.10.3', '10.10.10.1', '10.10.10.4']
    ckpt_client = CheckpointClient(config, hosts)

    source_shapes = get_source_shapes(config, source_shapes_dir)

    target_mp_rank = get_target_mp_rank(config, target_rank)
    if target_mp_rank is None:
        return

    target_rank_shapes = get_target_rank_shapes(target_shapes_dir, target_rank)

    target_state = get_rank0_struct(config, ckpt_client)

    # Fix integer keys being strings
    keys = list(
        target_state['optimizer']['optimizer_state_dict']['state'].keys())
    for key in keys:
        val = target_state['optimizer']['optimizer_state_dict']['state'][key]
        target_state['optimizer']['optimizer_state_dict']['state'][int(
            key)] = val
        del target_state['optimizer']['optimizer_state_dict']['state'][key]

    tensor_key_seqs = search_for_tensors(target_state)

    if tensor_key_seqs is None:
        raise ValueError("tensor_key_seqs is None")

    for tensor_key_seq in tensor_key_seqs:
        print(f'tranform {tensor_key_seq} for rank {target_mp_rank}')

        group_key = get_group_key(tensor_key_seq)

        new_tensor = repartition_optimiser_group_tensor(
            config, ckpt_client, source_shapes, target_rank_shapes,
            target_mp_rank, tensor_key_seq, group_key)

        target_size = get_target_size(target_rank_shapes[group_key])
        if new_tensor.size(0) != target_size:
            raise ValueError(f'new tensor {new_tensor.size(0)} '
                             f'!= target size {target_size}')

        set_tensor(target_state, tensor_key_seq, new_tensor)

    keys = search_for_not_primitive_Nor_tensor(target_state)
    if keys is None:
        raise ValueError('keys is None')
    file_name = "mp_rank_00_model_states.pt"
    for key in keys:
        key_str = [str(k) for k in key]
        path = '/'.join(key_str)
        path = os.path.join(file_name, path)
        value = ckpt_client.get_tensor(0, path)
        set_tensor(target_state, key, value)

    # Set sizes to target
    mp_world_size = config.target_mp_degree * config.target_pp_degree
    target_state['mp_world_size'] = mp_world_size
    target_state['dp_world_size'] = config.target_dp_degree

    # RNG state
    target_state["random_rng_state"][1] = tuple(
        target_state["random_rng_state"][1])
    target_state["random_rng_state"] = tuple(target_state["random_rng_state"])
    states = [
        'np_rng_state', 'torch_rng_state', 'cuda_rng_state',
        'rng_tracker_states'
    ]
    for state in states:
        target_state[state] = ckpt_client.get_tensor(
            0, os.path.join(file_name, state))

    target_state_name = f'mp_rank_{target_mp_rank:02d}_model_states.pt'
    target_state_path = os.path.join(config.output_dir, target_state_name)
    torch.save(target_state, target_state_path)


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

    transform_layers(config, args.target_rank)
    transform_state(config, args.target_rank)


if __name__ == '__main__':
    main()
