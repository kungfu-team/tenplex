import argparse

from .load import load


def main():
    parser = argparse.ArgumentParser(description='Write checkpoint')
    parser.add_argument('--device-rank', type=int)
    parser.add_argument('--mlfs-path', type=str)
    args = parser.parse_args()

    load(args.device_rank, args.mlfs_path)


if __name__ == '__main__':
    main()
