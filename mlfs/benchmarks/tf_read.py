#!/usr/bin/env python3.6
import argparse
import glob
import time

import tensorflow as tf
from tensorflow.python.util import deprecation
from tensorflow.keras import applications

deprecation._PRINT_DEPRECATION_WARNINGS = False

from dataset import create_dataset_from_files, create_random_input


class Log(object):
    def __init__(self):
        self.t0 = time.time()

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        d = time.time() - self.t0
        print('took %.3fs' % (d))


def read_lines(filename):
    return [line.strip() for line in open(filename)]


def show_pwd(root):
    fs = glob.glob(root + '/*')
    for f in fs:
        print(f)


def show_lines(ls):
    for l in ls:
        print(l)


def read_tf_record(filename):
    pass


def show_t(x):
    print(f'{x.dtype}{x.shape}')


def resnet50(xs):
    model = getattr(applications, 'ResNet50')(weights=None)
    logits = model(xs, training=True)
    return logits


def create_optimizer(learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate)
    # from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    # opt = SynchronousSGDOptimizer(opt)
    return opt


def create_model(xs=None, y_s=None):
    learning_rate = 0.1
    # xs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    # y_s = tf.placeholder(dtype=tf.int64, shape=[None])
    logits = resnet50(xs)
    pred = tf.argmax(logits, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(y_s, logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_s), tf.float32))
    opt = create_optimizer(learning_rate)
    train_op = opt.minimize(loss)
    return xs, y_s, train_op, acc, loss, logits


def parse_batch_sizes(mnt, prefix):
    def parse_bs(kv):
        k, v = kv.split(' ')
        return int(k), int(v)

    batch_sizes = [
        parse_bs(line.strip())
        for line in open(mnt + prefix + '/batch-sizes.txt')
    ]

    return batch_sizes


def run_train(ds, steps):
    samples, labels = ds
    xs, y_s, train_op, acc, loss, logits = create_model(samples, labels)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(steps):
            print(f'training step {step}')
            # feed_dict = {xs: samples}
            with Log():
                sess.run(train_op)


def run_task(mnt, dp_rank, root):
    batches = parse_batch_sizes(mnt, root)
    bs, count = batches[0]
    print(f'batch size: {bs} for {count} steps')
    filenames = read_lines(mnt + root + '/list.txt')
    filenames = [mnt + f for f in filenames]
    ds = create_dataset_from_files(filenames, bs)

    run_train(ds, count)
    # t0 = time.time()
    # with tf.Session() as sess:
    #     for step in range(count):
    #         x, y = sess.run(ds)
    #         print(f'rank {dp_rank} step {step}')
    #         # show_t(x)
    #         # show_t(y)
    #     print('finished')
    # d = time.time() - t0
    # print('{:.3f} samples/sec'.format(bs * count / d))


def run_job(mnt, root, args):
    head = read_lines(root + '/head.txt')[0]
    workers = read_lines(mnt + head)
    for i, w in enumerate(workers):
        if args.rank is None or args.rank == i:
            run_task(mnt, i, w)


def run(mnt, args):
    job_root = f'{mnt}/job/{args.job}'
    run_job(mnt, job_root, args)


def fake_run(args):
    bs = 32
    ds = create_random_input(bs)
    steps = 100
    run_train(ds, steps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fake-data', type=bool, default=False)
    p.add_argument('--job', type=str, default='0')
    p.add_argument('--rank', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.fake_data:
        fake_run(args)
    else:
        run('/mnt/mlfs', args)


main()
