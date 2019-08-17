import tensorflow as tf
import network
import numpy as np
import time

batch_size = 32
input_shape = 1024
layers = [1024, 1024]

num_samples = 100


# mlp
def build_network():
    input = tf.placeholder(tf.float32, shape=(batch_size, input_shape))

    for i, layer in enumerate(layers):
        h = tf.tanh(network.linear_custom(input, 'fc{}'.format(i), n_hidden=layer, histogram=True))

    return input, h


def run_session():
    input, h = build_network()

    init_all_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_all_op)

        start = time.time()
        for _ in range(num_samples):
            out = sess.run([h], feed_dict={
                input: np.random.rand(batch_size, input_shape)
            })

            # print(out)

        end = time.time()
        print(end - start)


if __name__ == '__main__':
    run_session()
