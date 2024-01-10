import sys

import torch


def show_dict(d):
    for k, v in sorted(d.items()):
        print('{:32} :: {}'.format(k, v.__class__))


def read_pt_file(filename):
    print(filename)
    d = torch.load(filename, map_location=torch.device('cpu'))
    show_dict(d)
    return
    #
    # optimizer_state_dict
    # param_shapes
    # ds_config
    # ds_version
    print(d['ds_version'])  #0.5.9+d93d924
    # /workspace/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/examples/ds_zero_stage_2_config.json
    print(d['ds_config'])
    print(d['param_shapes'].__class__)  # list
    for i, x in enumerate(d['param_shapes']):
        print(i)
        # print(x)
        for k, v in x.items():
            print(k)
            print(v)
            break
        print(len(x))

    # print(d['optimizer_state_dict'])
    print('optimizer_state_dict:')
    for k, v in d['optimizer_state_dict'].items():
        print(k)
        print(v.__class__)

    print('base_optimizer_state:')
    for s in d['optimizer_state_dict']['base_optimizer_state']:
        print(s)

    print('partition_count:')
    for s in d['optimizer_state_dict']['partition_count']:
        print(s)

    print('single_partition_of_fp32_groups:')
    for s in d['optimizer_state_dict']['single_partition_of_fp32_groups']:
        print(s)


def main(args):
    for filename in args:
        read_pt_file(filename)


main(sys.argv[1:])
