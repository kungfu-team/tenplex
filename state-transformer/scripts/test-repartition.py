import numpy as np
import torch

from hist import cmp_hist, hist


def create_hash_list(t, dim):
    assert 0 <= dim <= 2
    hash_list = []
    for i in range(t.shape[dim]):
        if dim == 0:
            h = hash(t[i])
        else:
            h = hash(t[:, i])
        hash_list.append(h)
    return hash_list


def test_hist():
    dim0 = 1024
    dim1 = 512

    one = torch.rand(dim0, dim1)

    two0 = one[:dim0//2]
    two1 = one[dim0//2:]

    hist_one = hist(one.numpy())
    hist_two0 = hist(two0.numpy())
    hist_two1 = hist(two1.numpy())

    hist_two_sum = hist_two0 + hist_two1

    cmp_hist(hist_one, hist_two_sum)


def test_hash():
    dim0 = 1024
    dim1 = 512

    one = torch.rand(dim0, dim1)

    two0 = one[:dim0//2]
    two1 = one[dim0//2:]

    hash_one = create_hash_list(one, 0)
    hash_two0 = create_hash_list(two0, 0)
    hash_two1 = create_hash_list(two1, 0)

    hash_two = hash_two0 + hash_two1

    print(hash_one == hash_two)


def main():
    test_hash()


if __name__ == '__main__':
    main()
