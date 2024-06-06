import argparse
from glob import glob
from os.path import join

import torch as pt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str)
    p.add_argument('--dataset', type=str)
    p.add_argument('--model', type=str)
    p.add_argument('--data-path', type=str)
    return p.parse_args()


def read_dataset(p):
    print(p)
    for f in glob(p + '/*'):
        print(f)


def main():
    args = parse_args()
    read_dataset(args.data_path)


main()
