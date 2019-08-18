import tensorflow as tf
import network
import numpy as np
import time

num_samples = 100


class Test:
    def __init__(self, layers, batch_size, input_shape, device='/cpu:0'):
        """
        - layers is list of dict
        e.g.
            [
                {'type': 'fc', 'size': '128', 'activation':'tanh'},
                {'type': 'fc', 'size': '256', 'activation':'tanh'},
            ]
        - batch_size is int
        - input_shape is int
        """
        self.batch_size = batch_size
        self.input_shape = input_shape

        # build network
        with tf.device(device):
            self.input, self.output = self.build_network(layers, batch_size, input_shape)

        init_all_op = tf.global_variables_initializer()
        self.sess = tf.Session()

        # initialize network
        self.sess.run(init_all_op)

    def build_network(self, layers, batch_size, input_shape):
        input = tf.placeholder(tf.float32, shape=(batch_size, input_shape))

        h = input
        for i, layer in enumerate(layers):
            # layer
            layer_type = layer['type']
            if layer_type == 'fc':
                h = network.linear_custom(h, 'fc{}'.format(i), n_hidden=layer['size'], histogram=True)
            elif layer['type'] == 'conv':
                raise NotImplemented
            else:
                raise NotImplemented

            # activation
            layer_activation = layer['activation']
            if layer_activation == 'tanh':
                h = tf.tanh(h)
            elif layer_activation == 'relu':
                h = tf.nn.relu(h)
            else:
                raise NotImplemented

        return input, h

    def run(self, input):

        print(input)

        start = time.time()

        out = self.sess.run([self.output], feed_dict={
            self.input: input
        })

        end = time.time()
        print(end - start)

    def close(self):
        self.sess.close()
        tf.reset_default_graph()


# if __name__ == '__main__':
#
#     test = Test()
#     test.run(np.random.rand(self.batch_size, self.input_shape))
#     test.close()


