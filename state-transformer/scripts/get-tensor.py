import json
import pickle

import requests


def get_tensor(path, r=None):
    url = 'http://127.0.0.1:21234' + '/' + path
    if r:
        payload = {
            'range': r,
        }
    else:
        payload = None
    result = requests.get(url, params=payload)
    tensor = pickle.loads(result.content)

    print(tensor.shape)


def get_struct(path):
    url = 'http://127.0.0.1:21234' + '/' + path
    result = requests.get(url)
    struct = json.loads(result.content)

    print(json.dumps(struct, indent=4))


def main():
    # tensor
    r = '[5:10]'
    path = 'global_step2000/mp_rank_00_model_states.pt/module/language_model/transformer/layers.0.attention.dense.weight'
    get_tensor(path, r)

    r = None
    path = 'global_step2000/zero_pp_rank_0_mp_rank_03_optim_states.pt/optimizer_state_dict/base_optimizer_state/0/exp_avg'
    get_tensor(path, r)

    # struct
    path = 'global_step2000/zero_pp_rank_0_mp_rank_03_optim_states.pt'
    get_struct(path)

    path = 'global_step2000/mp_rank_00_model_states.pt'
    get_struct(path)


if __name__ == '__main__':
    main()
