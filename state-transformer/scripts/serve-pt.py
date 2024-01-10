import argparse
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=21234)
    p.add_argument('--local-prefix', type=str)
    p.add_argument('--files', type=str)
    return p.parse_args()


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


def walk_pt_file(prefix, filename, f, state):
    print('walk_pt_file: {}'.format(prefix + filename))
    d = torch.load(prefix + filename, map_location=torch.device('cpu'))
    walk_pt(d, f, state, filename)


def mk_handler(prefix, filenames):
    print('prefix: {}'.format(prefix))
    for f in filenames:
        print('file: {}'.format(f))

    for f in filenames:
        if not f.startswith(prefix):
            raise '{} not start with {}'.format(f, prefix)

    filenames = [f[len(prefix):] for f in filenames]

    class Handler(BaseHTTPRequestHandler):

        def header(self, content_type='blob'):
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()

        def response_text(self, text):
            self.header()
            self.wfile.write(text.encode())

        def do_GET(self):
            print('GET ... {}'.format(self.path))
            lines = []

            def visit(o, lines, p):
                # print('visit: {}'.format(p))
                # print_pt(o, lines, p)
                lines += [p]

            for f in filenames:
                lines += [f]
                walk_pt_file(prefix, f, visit, lines)
                # break

            self.response_text(''.join([l + '\n' for l in lines]))

        def GET_list(self):
            pass

    return Handler


def runHTTP(port, server_class=HTTPServer, handler=BaseHTTPRequestHandler):
    print('http://127.0.0.1:%d/' % (port))
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler)
    httpd.serve_forever()


def run():
    args = parse_args()
    # for filename in args:
    #     print(filename)
    # read_pt_file(filename)
    runHTTP(args.port,
            handler=mk_handler(args.local_prefix, args.files.split(',')))


def main(args):
    # for i, a in enumerate(args):
    #     print('{} {}'.format(i, a))

    run()


main(sys.argv[1:])
