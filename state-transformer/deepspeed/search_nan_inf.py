import glob

import torch


def search_tensor(value):
    if isinstance(value, dict):
        for key, val in value.items():
            search_tensor(val)
    if isinstance(value, (list, set, tuple)):
        for val in value:
            search_tensor(val)
    if isinstance(value, torch.Tensor):
        if torch.isnan(value).all():
            print('is nan')
        if torch.isinf(value).all():
            print('is inf')


def main():
    base_dir = '/data/marcel/cont_pp1_mp2_dp1_200/0/ckpt/global_step100'
    files = glob.glob(base_dir + '/*')
    for fil in files:
        print(f'file {fil}')
        ckpt = torch.load(fil, map_location='cpu')
        search_tensor(ckpt)


if __name__ == '__main__':
    main()
