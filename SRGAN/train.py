import os
import tensorflow as tf
import math
import pdb
from srgan import SRGAN

initial_learning_rate = 0.01
epoch_step = 150
decay_factor = 0.1
batch_size = 16
height = 96
width = 96
channels = 3
dataset = '../lfw/train'
log_dir = './log'
model_dir = './model'


def train():
    x = tf.placeholder(tf.float32, [batch_size, height, width, channels])
    model = SRGAN(x=x, is_training=True, batch_size=batch_size, height=height, width=width)

    filename_list = get_image_paths(dataset)
    count = len(filename_list)
    loop = int(math.floor(count / batch_size))

    queue = tf.RandomShuffleQueue(10000, batch_size, tf.string, shapes=())
    init_queue = queue.enqueue_many((filename_list,))
    image = queue.dequeue()

    sess = tf.Session()
    with sess.as_default():
        init_queue.run()
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)
        rate = tf.train.exponential_decay(initial_learning_rate, global_step, epoch_step, decay_factor)
        g_train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(model.d_loss, var_list=model.d_variables)

        sess.run(tf.global_variables_initializer())

        # Restore the SRGAN network
        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary = tf.summary.merge_all()

        for step in xrange(loop):
            x_batch = [get_image(image.eval()) for i in xrange(batch_size)]
            run_metadata = tf.RunMetadata()
            summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            summary_str, g_loss, d_loss = sess.run([summary, g_train_op, d_train_op], feed_dict={x: x_batch})
            summary_writer.add_summary(summary_str, step)
            sess.run(update_global_step)
            print('step: %d' % (step + 1))

            if step % 20 == 0 and step != 0:
                checkpoint_file = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

        summary_writer.close()

    return d_train_op, g_train_op


def get_image_paths(train_dir):
    filename_list = os.listdir(train_dir)
    filename_list = [os.path.join(train_dir, filename_list[i]) for i in xrange(len(filename_list))]
    return filename_list


def get_image(image_path):
    print(image_path)
    file = tf.read_file(image_path)
    image = tf.image.decode_image(file, channels=channels)
    sess = tf.get_default_session()
    file, image = sess.run([file, image])
    return image


if __name__ == '__main__':
    train()

