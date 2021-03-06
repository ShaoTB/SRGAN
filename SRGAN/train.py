import os
import tensorflow as tf
from srgan import SRGAN
import time
import resource

initial_learning_rate = 0.00001
epoch_times = 10
decay_factor = 1
batch_size = 32
height = 96
width = 96
channels = 3
dataset = ['../lfw/train']
log_dir = './log'
model_dir = './model'


def train():
    x = tf.placeholder(tf.float32, [batch_size, height, width, channels])
    model = SRGAN(x=x, is_training=True, batch_size=batch_size, height=height, width=width)

    filename_list = get_image_paths(dataset)
    count = len(filename_list)
    loop = int(count / batch_size)
    queue = tf.RandomShuffleQueue(epoch_times * count, batch_size, tf.string, shapes=())
    init_queue = queue.enqueue_many((filename_list,))
    image = queue.dequeue()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)
        rate = tf.train.exponential_decay(initial_learning_rate, global_step, loop, decay_factor)
        g_train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(model.d_loss, var_list=model.d_variables)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()

        print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        for epoch in xrange(epoch_times):
            init_queue.run()
            while queue.size().eval() > batch_size:
                start_time = time.time()
                step = global_step.eval()
                print('step: %d' % step)
                x_batch = [get_image(image.eval()) for i in xrange(batch_size)]
                print("read image cost: %ds" % (time.time() - start_time))
                summary_str, d_loss, g_loss = sess.run([summary, d_train_op, g_train_op], feed_dict={x: x_batch})
                summary_writer.add_summary(summary_str, step)

                if global_step.eval() % 20 == 0:
                    checkpoint_file = os.path.join(model_dir, 'model.latest')
                    saver.save(sess, checkpoint_file)

                if global_step.eval() % 100 == 0:
                    summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                sess.run(update_global_step)
                print("step cost: %ds" % (time.time() - start_time))
                print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        summary_writer.close()

    return d_train_op, g_train_op


def get_image_paths(dir_list):
    filename_list = []
    for train_dir in dir_list:
        filename = os.listdir(train_dir)
        filename = [os.path.join(train_dir, filename[i]) for i in xrange(len(filename))]
        filename_list.extend(filename)
    return filename_list


def get_image(image_path):
    file = tf.read_file(image_path)
    image = tf.image.decode_image(file, channels=channels)
    sess = tf.get_default_session()
    file, image = sess.run([file, image])
    return image


if __name__ == '__main__':
    train()