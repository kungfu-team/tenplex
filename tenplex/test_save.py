import argparse

import torch

from .save import save


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--job-id', type=str)
    parser.add_argument('--step', type=str)
    parser.add_argument('--device-rank', type=int)
    parser.add_argument('--mlfs-path', type=str)
    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    save(ckpt, args.job_id, args.step, args.device_rank, args.mlfs_path,
         args.ip, args.port)


if __name__ == '__main__':
    main()
