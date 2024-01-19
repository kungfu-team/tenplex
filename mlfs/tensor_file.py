"""
APIs to access tensorfile in MLFS.
"""

import numpy as np
import requests

_dtypes = {
    'f32': np.float32,
    'f64': np.float64,
    #
    'i8': np.int8,
    'i16': np.int16,
    'i32': np.int32,
    'i64': np.int64,
    #
    'u8': np.uint8,
    'u16': np.uint16,
    'u32': np.uint32,
    'u64': np.uint64,
}


def _read_meta_file(name):
    lines = [line.strip() for line in open(name)]
    dt = _dtypes[lines[0]]
    rank = int(lines[2])
    dims = [int(d) for d in lines[2:2 + rank]]
    return dt, dims


def _read_data_file(name):
    return open(name, 'rb').read()


def read_tensor_file(name):
    """"Read tensor into a numpy array from MLFS."""
    dtype, shape = _read_meta_file(name + '.meta')
    data = _read_data_file(name)
    data = bytearray(data)
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _fmt_range(r):
    assert (r.step is None or r.step == 1)
    f = lambda d: '' if d is None else str(d)
    return f(r.start) + ':' + f(r.stop)


def query_tensor_file(path, rng):
    """"Read tensor slice into a numpy array from MLFS."""
    print(rng)
    host = '127.0.0.1'
    ctrl_port = 20010
    ranges = ','.join([_fmt_range(r) for r in rng])
    endpoint = 'http://{}:{}/query?path={}&range={}'.format(
        host,
        ctrl_port,
        path,
        ranges,
    )
    r = requests.get(endpoint)
    # print(r)
    if r.status_code != 200:
        print(r.reason)
    assert (r.status_code == 200)
    dtype = _dtypes[r.headers['x-tensor-dtype']]
    data = r.content
    shape = [int(d) for d in r.headers['x-tensor-dims'].split(',')]
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def upload_tensor(path, t):
    headers = {
        'Content-type': 'x-tensor',
    }
    host = '127.0.0.1'
    ctrl_port = 20010

    dims = [str(int(d)) for d in t.shape]
    endpoint = 'http://{}:{}/upload?dtype={}&dims={}&path={}'.format(
        host,
        ctrl_port,
        t.dtype,
        ','.join(dims),
        path,
    )
    r = requests.post(endpoint, headers=headers, data=t.tobytes())
    # print(r)
    assert (r.status_code == 200)
