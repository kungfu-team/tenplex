#!/usr/bin/env python3

import argparse
import os

import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.keras import applications

from imagenet import create_dataset, create_dataset_from_files
from logger import Logger


def parse_arsg():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='ResNet50')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--batch-number', type=int, default=256)
    p.add_argument('--log-period', type=int, default=16)
    p.add_argument('--num-warmup-batches', type=int, default=10)
    p.add_argument('--num-batches-per-iter', type=int, default=10)
    p.add_argument('--num-iters', type=int, default=10)
    p.add_argument('--no-cuda', action='store_true', default=False)
    p.add_argument('--mlfs-dir', type=str)
    p.add_argument('--data-dir', type=str)
    p.add_argument('--job', type=str)
    args = p.parse_args()
    args.cuda = not args.no_cuda
    return args


def log(s, nl=True):
    print(s, end='\n' if nl else '')


def get_data(args):
    if args.mlfs_dir:
        from tenplex.mlfs_path import MLFSPath
        p = MLFSPath(args.mlfs_dir)
        return create_dataset_from_files(p.filenames(args.job, 0),
                                         args.batch_size)
    elif args.data_dir:
        return create_dataset(args.data_dir, args.batch_size, 1024)
    else:
        data = tf.random_uniform([args.batch_size, 224, 224, 3])
        target = tf.random_uniform([args.batch_size, 1],
                                   minval=0,
                                   maxval=999,
                                   dtype=tf.int64)
        return data, target


def run(args, benchmark_step):
    log('Running warmup...')
    for _ in range(args.num_warmup_batches):
        benchmark_step()

    log('Running benchmark...')
    l = Logger()
    for _ in range(args.batch_number):
        benchmark_step()
        l.add(args.batch_size)
    l.report()


def build_train_op(args, data, target):
    model = getattr(applications, args.model)(weights=None)
    opt = tf.train.GradientDescentOptimizer(0.01)
    opt = hvd.DistributedOptimizer(opt)
    logits = model(data, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
    train_op = opt.minimize(loss)
    return train_op


def main():
    args = parse_arsg()
    hvd.init()

    config = tf.ConfigProto()
    if args.cuda:
        config.gpu_options.allow_growth = True
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config.gpu_options.visible_device_list = str(hvd.local_rank())
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config.gpu_options.allow_growth = False
        config.gpu_options.visible_device_list = ''

    data, target = get_data(args)
    train_op = build_train_op(args, data, target)
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as session:
        session.run(init)
        run(args, lambda: session.run(train_op))


main()
