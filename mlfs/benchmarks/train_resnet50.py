import argparse
from trace import Trace

from patient import Patient, Rate

with Trace('import tensorflow'):
    import tensorflow as tf
    from tensorflow.keras import applications
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    from dataset import create_dataset_from_files, create_random_input

from vfs import get_shard, remount, umount_mlfs


def resnet50(xs):
    model = getattr(applications, 'ResNet50')(weights=None)
    logits = model(xs, training=True)
    return logits


def create_optimizer(learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate)
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    opt = SynchronousSGDOptimizer(opt)
    return opt


def create_model(t):
    learning_rate = 0.1
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    y_s = tf.placeholder(dtype=tf.int64, shape=[None])
    logits = resnet50(xs)
    # t.log('logits: {}'.format(logits))
    pred = tf.argmax(logits, axis=1)
    # t.log('pred: {}'.format(pred))
    loss = tf.losses.sparse_softmax_cross_entropy(y_s, logits)
    # t.log('loss: {}'.format(loss))
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_s), tf.float32))
    # t.log('acc: {}'.format(acc))
    opt = create_optimizer(learning_rate)
    train_op = opt.minimize(loss)
    return xs, y_s, train_op, acc, loss, logits


def parse_args():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tfrecord-fs', type=str, default='')
    p.add_argument('--index-file', type=str, default='')
    p.add_argument('--global-batch-size', type=int, default='')
    p.add_argument('--prefix', type=str, default='')
    p.add_argument('--run-train-op', action='store_true', default=False)
    p.add_argument('--fake-data', action='store_true', default=False)
    return p.parse_args()


def show_tensor(x):
    return '{}[{}]'.format(x.dtype.name, ','.join(str(d) for d in x.shape))


def main():
    args = parse_args()

    if len(args.prefix) > 0:
        print('using disk fs prefix: {}'.format(args.prefix))
        shard = get_shard(args.prefix)
    else:
        prefix = remount(
            args.tfrecord_fs,
            args.index_file,
            args.global_batch_size,
            args.seed,
        )
        print('using vfs prefix: {}'.format(prefix))
        shard = get_shard(prefix)

    filenames = shard['filenames']
    batch_sizes = shard['batch_sizes']
    batch_size, step_count = batch_sizes[0]

    print('got {} filenames, batch size: {}, step: {}'.format(
        len(filenames),
        batch_size,
        step_count,
    ))

    with Trace('create_dataset') as t:
        if args.fake_data:
            ds = create_random_input(batch_size)
        else:
            ds = create_dataset_from_files(filenames, batch_size)

    with Trace('create_model') as t:
        xs, y_s, train_op, acc, loss, logits = create_model(t)

        init_op = tf.global_variables_initializer()

    with Trace('train') as t:
        with Patient() as p:
            with tf.Session() as sess:
                with Trace('run init_op') as t:
                    sess.run(init_op)

                with Rate('img', 'training') as r:
                    trace_step = False
                    for step in range(step_count):
                        with Trace('run data op', trace_step) as t:
                            samples, labels = sess.run(ds)
                        if (step + 1) % 100 == 0:
                            print(
                                'step: {}/{}, samples: {}, labels: {}'.format(
                                    step,
                                    step_count,
                                    show_tensor(samples),
                                    show_tensor(labels),
                                ))

                        if args.run_train_op:
                            with Trace('train op', trace_step) as t:
                                feed_dict = {
                                    xs: samples,
                                    y_s: labels,
                                }
                                sess.run(train_op, feed_dict)

                        p.yell('%5.02f%%' % (100.0 * (step + 1) / step_count))
                        #r.done(batch_size)
                        r.done(args.global_batch_size)
                r.all_done()

    with Trace('umount_mlfs') as t:
        # umount_mlfs()
        pass


with Trace('main') as t:
    main()
