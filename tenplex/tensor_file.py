"""
APIs to access tensorfile in MLFS.
"""

import numpy as np
import requests

_dtypes = {
    'f16': np.float16,
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
    with open(name, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    dt = _dtypes[lines[0]]
    rank = int(lines[2])
    dims = [int(d) for d in lines[2:2 + rank]]
    return dt, dims


def _read_data_file(name):
    with open(name, 'rb') as f:
        return f.read()


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


class TensorRequester:

    def __init__(self, ctrl_port, req_ip):
        self.ctrl_port = ctrl_port
        self.req_ip = req_ip

        # DEBUG
        #  self.total_upload = 0
        #  self.failed_upload = []

    def query_tensor_file(self, path, rang):
        """"Read tensor slice into a numpy array from MLFS."""
        ranges = ','.join([_fmt_range(r) for r in rang])
        endpoint = 'http://{}:{}/query?path={}&range={}'.format(
            self.req_ip,
            self.ctrl_port,
            path,
            ranges,
        )
        r = requests.get(endpoint)
        if r.status_code != 200:
            r.raise_for_status()
        dtype = _dtypes[r.headers['x-tensor-dtype']]
        data = r.content
        shape = [int(d) for d in r.headers['x-tensor-dims'].split(',')]
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def upload_tensor(self, path, t):
        headers = {
            'Content-type': 'x-tensor',
        }

        dims = [str(int(d)) for d in t.shape]
        endpoint = 'http://{}:{}/upload?dtype={}&dims={}&path={}'.format(
            self.req_ip,
            self.ctrl_port,
            t.dtype,
            ','.join(dims),
            path,
        )
        r = requests.post(endpoint, headers=headers, data=t.tobytes())
        if r.status_code != 200:
            r.raise_for_status()
            #  self.failed_upload.append(path)

        #  self.total_upload += 1
