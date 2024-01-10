import torch


def show_bytes(n):
    gi = 1 << 30
    mi = 1 << 20
    if n > gi:
        return '%.3fGiB' % (n / gi)
    elif n > mi:
        return '%.3fMiB' % (n / mi)
    else:
        return '{}'.format(n)


class Stat:

    def __init__(self, cnt=0, size=0):
        self.cnt = int(cnt)
        self.size = int(size)

    def __iadd__(self, a):
        self.cnt += a.cnt
        self.size += a.size
        return self

    def __str__(self):
        return '{} tensors, {} bytes ({})'.format(
            self.cnt,
            self.size,
            show_bytes(self.size),
        )


def show_pt(o, p=''):
    tot = Stat()
    if isinstance(o, dict):
        for k, v in sorted(o.items()):
            tot += show_pt(v, p + '/' + k)
    elif type(o) is list:
        for i, x in enumerate(o):
            tot += show_pt(x, p + '/' + str(i))
    elif type(o) is tuple:
        for i, x in enumerate(o):
            tot += show_pt(x, p + '/' + str(i))
    elif type(o) is torch.Tensor:
        print('{} :: {}{}'.format(p, str(o.dtype), list(o.shape)))
        tot += Stat(1, o.shape.numel() * o.element_size())
    elif type(o) is torch.Size:
        print('{} :: Shape{}'.format(p, list(o)))
    elif type(o) is int:
        print('{} :: {} | {}'.format(p, o.__class__, o))
    elif type(o) is bool:
        print('{} :: {} | {}'.format(p, o.__class__, o))
    elif type(o) is float:
        print('{} :: {} | {}'.format(p, o.__class__, o))
    else:
        print('{} :: {}'.format(p, o.__class__))

    return tot


def walk_pt(o, f, state, p):
    if isinstance(o, dict):
        for k, v in sorted(o.items()):
            walk_pt(v, f, state, p + '/' + k)
    elif type(o) is list:
        for i, x in enumerate(o):
            walk_pt(x, f, state, p + '/' + str(i))
    elif type(o) is tuple:
        for i, x in enumerate(o):
            walk_pt(x, f, state, p + '/' + str(i))
    else:
        f(o, state, p)


def print_pt(o, _, p):
    if type(o) is torch.Tensor:
        print('{} :: {}{}'.format(p, str(o.dtype), list(o.shape)))
    elif type(o) is torch.Size:
        print('{} :: Shape{}'.format(p, list(o)))
    elif type(o) is int:
        print('{} :: {} | {}'.format(p, o.__class__, o))
    elif type(o) is bool:
        print('{} :: {} | {}'.format(p, o.__class__, o))
    elif type(o) is float:
        print('{} :: {} | {}'.format(p, o.__class__, o))
    else:
        print('{} :: {}'.format(p, o.__class__))


def get_tree(o):
    """Export the tree structure of a .pt file."""
    pass


def read_pt_file(filename):
    print('filename : {}'.format(filename))
    d = torch.load(filename, map_location=torch.device('cpu'))
    tot = show_pt(d)
    print('{}'.format(tot))
