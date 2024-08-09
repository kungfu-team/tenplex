import argparse


def add_tenplex_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Tenplex")

    group.add_argument("--tenplex", action="store_true")
    group.add_argument("--mlfs-path", type=str, default=None)
    group.add_argument("--jobid", type=str, default=None)
    group.add_argument("--host-ip", type=str, default=None)
    group.add_argument("--mlfs-port", type=int, default=None)
    group.add_argument("--scheduler-addr", type=str, default=None)
    group.add_argument("--tenplex-train-iters", type=int, default=None)
    group.add_argument("--gen-para-config", action="store_true")

    return parser
