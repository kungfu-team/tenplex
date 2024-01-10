import torch
from collections import Counter
# from patient import Patient


def show_hist(cnt):
    for (b, c) in enumerate(cnt):
        if c > 0:
            print('{:8} occurance of {:3}'.format(c, b))


def cmp_hist(a, b, prefix=''):
    diff = 0
    for i, (m, n) in enumerate(zip(a, b)):
        if m != n:
            #  print('{} diff #{:6} different count of byte 0x{:02x}, {} != {}'.
                  #  format(
                      #  prefix,
                      #  diff,
                      #  i,
                      #  m,
                      #  n,
                  #  ))
            diff += 1

    return diff


def hist(x):
    bs = x.tobytes()
    c = Counter(bs)
    cnt = [0 for _ in range(256)]
    for b, c in c.items():
        cnt[b] = c
    return cnt


def test():

    x = torch.Tensor([1.0, 2.0])
    y = torch.Tensor([1.0, 3.0])

    h1 = hist(x)
    show_hist(h1)
    h2 = hist(y)
    show_hist(h2)

    cmp_hist(h1, h2)


# test()
