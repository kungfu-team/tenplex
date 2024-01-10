import sys

import torch


def show_pt(o, p='/'):
    if isinstance(o, dict):
        for k, v in sorted(o.items()):
            show_pt(v, p + '/' + k)
    else:
        print('{} :: {}'.format(p, o.__class__))


def show_dict(d):
    for k, v in sorted(d.items()):
        print('{:32} :: {}'.format(k, v.__class__))


def read_pt_file(filename):
    print(filename)
    d = torch.load(filename, map_location=torch.device('cpu'))
    show_pt(d)

    # show_dict(d)

    # print('{} buffer_names'.format(len(d['buffer_names'])))
    # for i, s in enumerate(d['buffer_names']):
    #     print('{:6} {}'.format(i, s))
    # print('')

    # print('lr_scheduler:')
    # show_dict(d['lr_scheduler'])
    # print('')

    # print('module:')
    # show_dict(d['module'])
    # print('')

    # print('language_model:')
    # show_dict(d['module']['language_model'])
    # print('')

    # print('embedding:')
    # show_dict(d['module']['language_model']['embedding'])
    # print('')
    # print('transformer:')
    # show_dict(d['module']['language_model']['transformer'])
    # print('')

    #
    # optimizer_state_dict
    # param_shapes
    # ds_config
    # ds_version


def main(args):
    for filename in args:
        read_pt_file(filename)
        print('')


main(sys.argv[1:])
