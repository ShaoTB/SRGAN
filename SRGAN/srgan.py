from layer import *
from VGG19.vgg19 import *

vgg = Vgg19()


class SRGAN:
    def __init__(self, x, is_training=False, batch_size=32, height=96, width=96, channels=3):
        self.K = 4
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.channels = channels
        self.x = x
        self.is_training = is_training
        self.downscaled = self.downscale(self.x)
        self.imitation = self.generator(self.downscaled, self.is_training, False)
        tf.summary.image('fake_image', self.imitation)
        self.true_output = self.discriminator(self.x, self.is_training, False)
        self.fake_output = self.discriminator(self.imitation, self.is_training, True)
        self.g_loss, self.d_loss = self.inference_losses(self.x, self.imitation, self.true_output, self.fake_output)

    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [3, 3, 64, 3], [self.batch_size, self.height / 4, self.width / 4, 64], 1, 'deconv1')
            x = tf.nn.relu(x)
            shortcut = x
            for i in range(5):
                mid = x
                with tf.variable_scope('block{}a'.format(i+1)):
                    x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, self.height / 4, self.width / 4, 64], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('block{}b'.format(i+1)):
                    x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, self.height / 4, self.width / 4, 64], 1)
                    x = batch_normalize(x, is_training)
                x = tf.add(x, mid)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, self.height / 4, self.width / 4, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.add(x, shortcut)
            with tf.variable_scope('deconv3'):
                x = deconv_layer(x, [3, 3, 256, 64], [self.batch_size, self.height / 4, self.width / 4, 256], 1)
                x = pixel_shuffle_layer(x, 2, 64)  # n_split = 256 / 2 ** 2
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv4'):
                x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, self.height / 2, self.width / 2, 64], 1)
                x = pixel_shuffle_layer(x, 2, 16)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv5'):
                x = deconv_layer(x, [3, 3, 3, 16], [self.batch_size, self.height, self.width, 3], 1)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x

    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            x = flatten_layer(x)
            with tf.variable_scope('fc'):
                x = full_connection_layer(x, 1024)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1)

        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x

    def downscale(self, x):
        downscaled = tf.image.resize_images(x, [self.height / self.K, self.width / self.K], method=tf.image.ResizeMethod.BICUBIC)
        return downscaled

    def inference_losses(self, x, imitation, true_output, fake_output):
        def inference_content_loss(x, imitation):
            vgg.build(x)
            x_phi = vgg.conv5_4
            tf.summary.histogram('original_phi', x_phi)
            vgg.build(imitation)
            imitation_phi = vgg.conv5_4
            tf.summary.histogram('generator_phi', imitation_phi)
            content_loss = tf.reduce_mean(tf.square(x_phi - imitation_phi))
            tf.summary.scalar('content_loss', content_loss)
            return content_loss

        def inference_adv_loss(true_output, fake_output):
            tf.summary.histogram('sample_discrimination', true_output)
            tf.summary.histogram('generator_discrimination', fake_output)
            alpha = 1e-3
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output)))
            d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=true_output, labels=tf.ones_like(true_output)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output)))
            d_loss = (d_loss_true + d_loss_fake) / 2

            tf.summary.scalar('generator_loss', g_loss)
            tf.summary.scalar('discriminator_loss_sample', d_loss_true)
            tf.summary.scalar('discriminator_loss_generator', d_loss_fake)
            return g_loss * alpha, d_loss

        content_loss = inference_content_loss(x, imitation)
        generator_loss, discriminator_loss = inference_adv_loss(true_output, fake_output)
        g_loss = content_loss + generator_loss
        d_loss = discriminator_loss
        return g_loss, d_loss
