import os

import numpy as np

from tensor_file import query_tensor_file, read_tensor_file, upload_tensor


def test_1():
    x = read_tensor_file('a')
    print(x)


def test_2():
    # x = T()
    # x[1:2, 3:4, :]
    x = query_tensor_file('a', [slice(), slice()])
    print(x)


usr = os.getenv('USER')


def test_upload_with(x, path):
    print(x)
    upload_tensor(path, x)
    mnt = f'/data/{usr}/mlfs'
    y = read_tensor_file(mnt + path)
    print(y)
    # assert (x .eq() y)


def test_upload():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    test_upload_with(x, '/x')

    x = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6]], dtype=np.int8)
    test_upload_with(x, '/y')


def test_query():
    data = list(range(16))
    x = np.array(data, dtype=np.float32).reshape((4, 4))
    test_upload_with(x, '/x')

    y = query_tensor_file('/x', [slice(1, 3), slice(1, 3)])
    print(y)

    y = query_tensor_file('/x', [slice(None), slice(None)])
    print(y)

    y = query_tensor_file('/x', [slice(1, 3)])
    print(y)

    y = query_tensor_file('/x', [slice(None), slice(1, 3)])
    print(y)


# test_1()
# test_2()

# test_upload()
test_query()
