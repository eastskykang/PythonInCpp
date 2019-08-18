import tensorflow as tf
import network
import numpy as np
import time

batch_size = 32
input_shape = 1024
layers = [1024, 1024]

num_samples = 100


class Test:
    def __init__(self, layers, batch_size, input_shape, device='/device:GPU:0'):

        # build network
        self.input, self.output = self.build_network(layers, batch_size, input_shape)

        init_all_op = tf.global_variables_initializer()
        self.sess = tf.Session()

        # initialize network
        self.sess.run(init_all_op)

    def build_network(self, layers, batch_size, input_shape):
        """
        layers is list of dict

        e.g.
            [{'type': 'fc', 'size': '128', 'activation':'tanh'}, {'type': 'fc', 'size': '256', 'activation':'tanh'}]
        """
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
                raise NotImplemented
            else:
                raise NotImplemented

        return input, h

    def run(self):
        start = time.time()
        for _ in range(num_samples):
            out = self.sess.run([self.output], feed_dict={
                input: np.random.rand(batch_size, input_shape)
            })

            # print(out)

        end = time.time()
        print(end - start)


# if __name__ == '__main__':
#     run_session()
