import tensorflow as tf
import network
import numpy as np
import time

batch_size = 32
input_shape = 128

num_samples = 100

# mlp
def build_network():
    input = tf.placeholder(tf.float32, shape=(batch_size, input_shape))
    h = tf.tanh(network.linear_custom(input, 'fc1', n_hidden=128, histogram=True))
    h = tf.tanh(network.linear_custom(h, 'fc2', n_hidden=128, histogram=True))

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
