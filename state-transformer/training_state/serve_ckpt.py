import argparse
import collections
import json
import os
import pickle
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=21234)
    p.add_argument('--data', type=str, default='./data')
    return p.parse_args()


def parse_slice(slice_str):
    no_space = slice_str.replace(' ', '')
    no_square_brackets = no_space.replace('[', '')
    no_square_brackets = no_square_brackets.replace(']', '')
    dimensions = no_square_brackets.split(',')

    slices = []
    for dim in dimensions:
        boarders = dim.split(':')
        parsed_boarders = []
        for boarder in boarders:
            if boarder != '':
                parsed_boarders.append(int(boarder))
            else:
                parsed_boarders.append(None)
        slices.append(parsed_boarders)

    return slices


def is_empty_or_none(string):
    if string:
        if string == '':
            return True
        return False
    return True


def slice_tensor(tensor, slice_str):
    if len(slice_str) == 0:
        return tensor
    slices = parse_slice(slice_str)

    if len(tensor.shape) < len(slices):
        print(
            f'more slices {len(slices)} than tensor dimensions {len(tensor.shape)}'
        )

    a = slices[0][0]
    b = slices[0][1]

    tensor_shape = tensor.shape
    if not is_empty_or_none(a) and a >= tensor_shape[0]:
        raise ValueError(
            f'Lower bound {a} is greater or equal to tensor\'s first dimension {tensor_shape[0]}'
        )
    if not is_empty_or_none(b) and b > tensor_shape[0]:
        raise ValueError(
            f'Upper bound {b} is greater than tensor\'s first dimension {tensor_shape[0]}'
        )

    if len(slices) == 1:

        if a == '':
            tensor_slice = tensor[:b]
        elif b == '':
            tensor_slice = tensor[a:]
        else:
            tensor_slice = tensor[a:b]
    elif len(slices) == 2:
        c = slices[1][0]
        d = slices[1][1]

        if not is_empty_or_none(c) and c >= tensor_shape[1]:
            raise ValueError(
                f'Lower bound {c} is greater or equal to tensor\'s second dimension {tensor_shape[1]}'
            )
        if not is_empty_or_none(d) and d > tensor_shape[1]:
            raise ValueError(
                f'Upper bound {d} is greater than tensor\'s second dimension {tensor_shape[1]}'
            )

        if a == '':
            if c == '':
                tensor_slice = tensor[:b, :d]
            elif d == '':
                tensor_slice = tensor[:b, c:]
            else:
                tensor_slice = tensor[:b, c:d]
        elif b == '':
            if c == '':
                tensor_slice = tensor[a:, :d]
            elif d == '':
                tensor_slice = tensor[a:, c:]
            else:
                tensor_slice = tensor[a:, c:d]
        else:
            if c == '':
                tensor_slice = tensor[a:b, :d]
            elif d == '':
                tensor_slice = tensor[a:b, c:]
            else:
                tensor_slice = tensor[a:b, c:d]
    else:
        raise NotImplementedError('slices length {len(slices)} is too long')

    tensor_copy = torch.tensor(tensor_slice)
    return tensor_copy


def get_tensor(ckpt, keys):
    value = ckpt[keys[0]]
    for key in keys[1:]:
        try:
            int_key = int(key)
            try:
                value = value[int_key]
            except KeyError:
                raise KeyError(f'key {int_key} of {keys} is not in checkpoint')
        except ValueError:
            value = value[key]
    return value


def create_value_dict(value):
    if value is None:
        return None
    if isinstance(value, (collections.OrderedDict, dict)):
        elements = {}
        for key, val in value.items():
            elements[key] = create_value_dict(val)
        return elements
    if isinstance(value, torch.Tensor):
        return {'tensor': list(value.size())}
    if isinstance(value, (list, tuple)):
        elements = []
        for val in value:
            elements.append(create_value_dict(val))
        return elements
    if isinstance(value, (int, float, str, bool)):
        return value

    return 'NotPrimitiveNorTensor'


def create_ckpt_dict(ckpt):
    elements = {}
    for key, val in ckpt.items():
        elements[key] = create_value_dict(val)
    return elements


def generate_ckpt_handler(args):
    ckpts = {}

    class TensorHandler(BaseHTTPRequestHandler):

        def header(self, content_type='text/plain'):
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()

        def ret_txt(self, path):
            file_path = os.path.join(args.data, path)
            with open(file_path, 'r') as txt_file:
                txt = txt_file.read()

            self.header()
            self.wfile.write(txt.encode())

        def ret_tensor(self, url, path):
            parts = path.split('.pt')
            filename = os.path.join(args.data, parts[0] + '.pt')
            ckpt = get_ckpt(filename, ckpts)
            sec_part = parts[1][1:]
            keys = sec_part.split('/')

            tensor = get_tensor(ckpt, keys)

            params = parse_qs(url.query)
            r = params.get('range', [''])[0]
            tensor_slice = slice_tensor(tensor, r)

            self.header()
            self.wfile.write(pickle.dumps(tensor_slice))

        def ret_struct(self, path):
            parts = path.split('.pt')
            filename = os.path.join(args.data, parts[0] + '.pt')
            ckpt = get_ckpt(filename, ckpts)

            ckpt_dict = create_ckpt_dict(ckpt)
            ckpt_json = json.dumps(ckpt_dict)

            self.header()
            self.wfile.write(ckpt_json.encode())

        def ret_ckpt(self, path):
            filename = os.path.join(args.data, path)
            ckpt = get_ckpt(filename, ckpts)
            ckpt_dict = create_ckpt_dict(ckpt)

            self.header()
            self.wfile.write(pickle.dumps(ckpt_dict))

        def do_GET(self):
            if self.path == '/quit':
                self.header()
                self.wfile.write(b'OK\n')
                th = threading.Thread(target=self.server.shutdown)
                th.start()
                return

            url = urlparse(self.path)
            path = url.path
            if path.startswith('/'):
                path = path[1:]

            if path.startswith('latest'):
                self.ret_txt(path)
            elif path.endswith('.pt'):
                self.ret_struct(path)
                #  self.ret_ckpt(path)
            else:
                self.ret_tensor(url, path)

        def GET_list(self):
            pass

        def log_message(self, format, *args):
            pass

    return TensorHandler


def runHTTP(port, handler=BaseHTTPRequestHandler):
    print('http://localhost:%d/' % (port))
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, handler)
    httpd.serve_forever()


def get_ckpt(filename, ckpts):
    if filename in ckpts:
        return ckpts[filename]
    else:
        ckpts[filename] = torch.load(filename,
                                     map_location=torch.device('cpu'))
        return ckpts[filename]


def main():
    args = parse_args()
    runHTTP(args.port, handler=generate_ckpt_handler(args))


if __name__ == '__main__':
    main()
