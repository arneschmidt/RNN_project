import tensorflow as tf


def deepnn(x_image, x_prev_seg, x_dir):
    x_image = tf.reshape(x_image, [-1, 60, 80, 1])
    x_prev_seg = tf.reshape(x_prev_seg, [-1, 60, 80, 1])
    x_dir = tf.reshape(x_dir, [-1, 1, 1, 1])
    x_pre_dir = tf.multiply(tf.ones_like(x_prev_seg), x_dir*127)
    x_pre = tf.concat([x_prev_seg, x_pre_dir], axis=3)

    conv1_image = tf.layers.conv2d(
        inputs=x_image,
        filters=5,
        kernel_size=[5, 5],
        use_bias=True,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.1),
        bias_initializer=tf.constant_initializer(0),
        activation=tf.nn.relu)

    conv1_seg = tf.layers.conv2d(
        inputs=x_pre,
        filters=5,
        kernel_size=[5, 5],
        use_bias=True,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.1),
        bias_initializer=tf.constant_initializer(0),
        activation=tf.nn.relu)

    concat = tf.concat([conv1_image, conv1_seg], axis=3)

    conv3 = tf.layers.conv2d(
        inputs=concat,
        filters=5,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=3,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=10),
        bias_initializer=tf.constant_initializer(120),
        activation=tf.nn.relu)

    hidden_layers = [x_image, conv1_image, x_pre, conv1_seg, conv3, conv4, conv5]
    y_out = tf.reshape(conv5, [-1, 60, 80])

    return y_out, hidden_layers

