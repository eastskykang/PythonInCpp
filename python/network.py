import tensorflow as tf


def linear_custom(input_tensor, scope, n_hidden, *,
                  init_bias=0.0, init_scale=0.01, initializer=tf.contrib.layers.xavier_initializer, histogram=False):

    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        if initializer is tf.contrib.layers.xavier_initializer:
            weight = tf.get_variable("w", [n_input, n_hidden], initializer=initializer())
        else:
            weight = tf.get_variable("w", [n_input, n_hidden], initializer=initializer(init_scale))
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))

        if histogram:
            tf.summary.histogram(name='{}/w'.format(scope), values=weight)
            tf.summary.histogram(name='{}/b'.format(scope), values=bias)

        return tf.matmul(input_tensor, weight) + bias


def conv_custom(input_tensor, scope, *, n_filters, filter_size, stride,
                pad='VALID', data_format='NHWC', one_dim_bias=False, init_scale=0.01,
                initializer=tf.contrib.layers.xavier_initializer, trainable=True, histogram=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_size, filter_size, n_input, n_filters]
    with tf.variable_scope(scope):
        if initializer is tf.contrib.layers.xavier_initializer:
            weight = tf.get_variable("w", wshape, initializer=initializer(), trainable=trainable)
        else:
            weight = tf.get_variable("w", wshape, initializer=initializer(init_scale), trainable=trainable)
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0), trainable=trainable)
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)

        if histogram:
            tf.summary.histogram(name='{}/w'.format(scope), values=weight)
            tf.summary.histogram(name='{}/b'.format(scope), values=bias)

        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)
