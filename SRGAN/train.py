import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pdb

from srgan import SRGAN

learning_rate = 1e-3
batch_size = 32
dataset = '../lfw/train'

def train():
    model = SRGAN(is_training=True)

    filenames = get_image_paths(dataset)
    count = len(filenames)

    sess = tf.Session()
    queue = tf.RandomShuffleQueue(10000, batch_size, tf.string)
    sess.run(queue.enqueue_many(tf.cast(filenames, tf.string)))

    g_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.g_loss, var_list=model.g_variables)
    d_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.d_loss, var_list=model.d_variables)

    sess.run(tf.global_variables_initializer())

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

    loop = int(math.floor(count / batch_size))
    for step in xrange(loop):
        images = queue.dequeue_many(batch_size)
        x_batch = get_images(images)
        x_batch = tf.image.resize_images(x_batch, [224, 224], tf.image.ResizeMethod.BICUBIC)
        pdb.set_trace()
        sess.run([g_train_op, d_train_op], feed_dict={model.x: x_batch})
        if step % 50 == 0:
            print('step:' % step)

def get_image_paths(dataset):
    filenames = os.listdir(dataset)
    for i in xrange(len(filenames)):
        filenames[i] = dataset + '/' + filenames[i]
    return filenames

def get_images(image_paths):
    return tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(image_paths), channels=3), dtype=tf.uint8)

if __name__ == '__main__':
    train()

