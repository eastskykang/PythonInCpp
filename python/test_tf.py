import tensorflow as tf
import numpy as np
import yaml
import time
import network

num_samples = 100


class Test:
    def __init__(self, layers, batch_size, input_shape, device='/cpu:0', intra_thread=0, inter_thread=0):
        """
        - layers: list of dict
        e.g.
            [
                {'type': 'fc', 'size': '128', 'activation':'tanh'},
                {'type': 'fc', 'size': '256', 'activation':'tanh'},
            ]
        - batch_size: int
        - input_shape: int
        - intra_thread: Nodes that can use multiple threads to parallelize their execution will schedule the individual pieces into this pool.
        - inter_thread: All ready nodes are scheduled in this pool.
        """
        self.batch_size = batch_size
        self.input_shape = input_shape

        # build network
        with tf.device(device):
            self.input, self.output = self.build_network(layers, batch_size, input_shape)

        init_all_op = tf.global_variables_initializer()

        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = intra_thread
        config.inter_op_parallelism_threads = inter_thread
        self.sess = tf.Session(config=config)

        # initialize network
        self.sess.run(init_all_op)

        # run with dummy (for warm up)
        self.sess.run([self.output], feed_dict={
            self.input: np.random.rand(batch_size, input_shape)
        })

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
            elif layer_activation == 'none':
                pass
            else:
                raise NotImplemented

        return input, h

    def run(self, input):
        return self.sess.run([self.output], feed_dict={
            self.input: input
        })

    def close(self):
        self.sess.close()
        tf.reset_default_graph()


if __name__ == '__main__':

    with open("/home/donghok/git/tfbench/yaml/test.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        test_specs = config['tests']

        for test_spec in test_specs:

            tag = test_spec['tag']
            device = test_spec['device']
            batch_size = test_spec['batch']
            input_size = test_spec['input']
            layers = test_spec['layers']
            num_steps = test_spec['step']

            test = Test(layers, batch_size, input_size, device)

            input = np.random.rand(batch_size, input_size)

            start = time.time()
            for _ in range(num_steps):
                output = test.run(input)
            end = time.time()

            print('tag          : {}'.format(tag))
            print('step         : {}'.format(num_steps))
            print('elapsed time : {}'.format(end-start))

            test.close()


