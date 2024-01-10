import sys
from . import read_pt_file


def main(args):
    for filename in args:
        read_pt_file(filename)


if __name__ == '__main__':
    main(sys.argv[1:])

# python3  -m training_state
