# -*- coding: utf-8 -*-
# [Unit tests](https://www.tensorflow.org/api_guides/python/test)

import tensorflow as tf
import os
import time

from context import tfcifarclassify
from tfcifarclassify.modelloader.imagenet import resnet
from tfcifarclassify.dataloader import cifar_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', os.path.expanduser('~/Data/cifar-10-batches-bin/data_batch_1.bin'), 'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', os.path.expanduser('~/Data/cifar-10-batches-bin/data_batch_1.bin'), 'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50, 'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False, 'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/tmp/{}'.format(time.time()), 'Directory to keep the checkpoints. Should be a parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used for training. (0 or 1)')


def main(_):
    dev = '/cpu:0'
    batch_size = 1
    hps = resnet.HParams(batch_size=batch_size,
                         num_classes=10,
                         min_lrn_rate=0.0001,
                         lrn_rate=0.1,
                         num_residual_units=5,
                         use_bottleneck=False,
                         weight_decay_rate=0.0002,
                         relu_leakiness=0.1,
                         optimizer='mom')
    with tf.device(dev):
        images, labels = cifar_input.build_input('cifar10', FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
        # print('images:', images)
        # print('labels:', labels)
        model = resnet.ResNet(hps, images, labels, FLAGS.mode)
        model.build_graph()

        truth = tf.argmax(model.labels, axis=1)
        predictions = tf.argmax(model.predictions, axis=1)
        precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': model.global_step, 'loss': model.cost, 'precision': precision}, every_n_iter=100)

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_root, hooks=[logging_hook], save_summaries_steps=0, config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(model.train_op)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
