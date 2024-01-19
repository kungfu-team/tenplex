#!/usr/bin/env python3

import argparse
import glob
import os
from os.path import join

import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def list_tags(filename):
    it = tf.train.summary_iterator
    tags = dict()
    for e in it(filename):
        # print(e)
        for i, v in enumerate(e.summary.value):
            # print('{} {}'.format(i, v))
            if v.tag not in tags:
                tags[v.tag] = 0

            tags[v.tag] += 1

    if False:
        items = list(sorted([(c, k) for k, c in tags.items()]))
        for c, k in items:
            print('- {:8} {}'.format(c, k))

    return list(tags.keys())


def get_t0(filename):
    it = tf.train.summary_iterator
    for idx, e in enumerate(it(filename)):
        return e.wall_time
    return 0


def extract(filenames, t0, output, tag='lm loss'):
    it = tf.train.summary_iterator
    with open(output, 'w') as f:
        for filename in filenames:
            print(filename)
            # tags = list_tags(filename)
            # print(tags)
            # for idx, t in enumerate(tags):
            #     print(f'- {idx} {t}')
            for idx, e in enumerate(it(filename)):
                # print(e.wall_time)
                for i, v in enumerate(e.summary.value):
                    # s = str(v).replace('\n', ' ')
                    # print(f'- {idx} {i}: {s}')
                    if v.tag == tag:
                        # print(v.simple_value)
                        # pass
                        f.write('{} {}\n'.format(e.wall_time - t0,
                                                 v.simple_value))


def find_tf_events(prefix):
    print(prefix)
    return glob.glob(join(prefix, '**/events.out.tfevents*'), recursive=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir
    for f in find_tf_events(log_dir):
        print(f)
    job_1 = 'job-1'
    job_2 = 'job-2'

    a = find_tf_events(join(log_dir, f'training/{job_1}'))
    b = find_tf_events(join(log_dir, f'training/{job_2}'))

    # print(a)
    # print(b)
    # return
    filenames = a + b
    t0 = min([get_t0(f) for f in filenames])

    print(t0)
    extract(a, t0, 'a.log')
    extract(b, t0, 'b.log')


main()
