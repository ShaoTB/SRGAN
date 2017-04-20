import os
import tensorflow as tf
import math
import pdb

from srgan import SRGAN

learning_rate = 1e-3
batch_size = 32
dataset = '/media/inverse/Document/eva'
log_dir = './backup'

def train():
    model = SRGAN(is_training=True)

    filenames = get_image_paths(dataset)
    count = len(filenames)

    queue = tf.RandomShuffleQueue(10000, batch_size, tf.string)
    init_queue = queue.enqueue_many((filenames,))
    imagebatch = queue.dequeue_many(batch_size)
    sess = tf.Session()
    with sess.as_default():
        init_queue.run()
        pdb.set_trace()
        g_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.d_loss, var_list=model.d_variables)

        sess.run(tf.global_variables_initializer())

        # Restore the SRGAN network
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()

        loop = int(math.floor(count / batch_size))
        for step in xrange(loop):

            x_batch = get_images(imagebatch.eval())
            sess.run([g_train_op, d_train_op], feed_dict={model.x: x_batch})
            summary_str = sess.run(summary)
            summary_writer.add_summary(summary_str, step)
            if step % 10 == 0:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('step:' % step)

def get_image_paths(dataset):
    filenames = os.listdir(dataset)
    for i in xrange(len(filenames)):
        filenames[i] = dataset + '/' + filenames[i]
    return filenames

def get_images(image_paths):
    batch = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(image_paths), channels=3), dtype=tf.uint8)
    return tf.image.resize_images(batch, [224, 224], tf.image.ResizeMethod.BICUBIC)

if __name__ == '__main__':
    train()

