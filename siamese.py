import tensorflow as tf

tf.set_random_seed(0)

LEARNING_RATE = 0.01

class Siamese:
    # Set up siamese network
    def __init__(self):
        # Variables to hold data later to feed into computation graph
        # Assign it a shape of [None, 784], where 784 is the dimensionality
        # of a single flattened 28 by 28 pixel MNIST image, and None indicates 
        # that the first dimension, corresponding to the batch size, can be of any size.
        # First input image
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        # Second input image
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        # Label of the image paired
        # 1: paired, 0: unpaired
        self.y = tf.placeholder(tf.float32, [None])
        
        # Parallel streams for siamese network
        with tf.variable_scope('siamese') as scope:
            self.output1 = self.createStream(self.x1)
            # Share weights
            scope.reuse_variables()
            self.output2 = self.createStream(self.x2)
        
        # Loss
        self.loss = self.lossFunc()

        # Optimizer
        self.optimizer = self.initOptimizer()

    def createStream(self, x):
        # Layer vector
        fc_layer1 = self.createLayer(x, 1024, 'fc_layer1')
        # Add non-linearity to fetch to next layer
        ac1 = tf.nn.relu(fc_layer1)
        
        # 2 other layers
        fc_layer2 = self.createLayer(ac1, 1024, 'fc_layer2')
        ac2 = tf.nn.relu(fc_layer2)
        fc_layer3 = self.createLayer(ac2, 2, 'fc_layer3')
        
        return fc_layer3

    def createLayer(self, tf_input, n_hidden_units, name):
        assert len(tf_input.get_shape()) == 2
        n_features = tf_input.get_shape()[1]
        
        # Define Initializer to generate a truncated normal distribution
        # with standard deviation = 0.01
        initializer_weight = tf.truncated_normal_initializer(stddev=0.01)
        initializer_bias = tf.constant_initializer(0.01)
        
        # Tensors
        W = tf.get_variable(
            name = name+'_W',
            dtype = tf.float32,
            shape = [n_features, n_hidden_units],
            initializer = initializer_weight)
        b = tf.get_variable(
            name = name + '_b',
            dtype = tf.float32,
            shape = [n_hidden_units],
            initializer = initializer_bias)
        
        # Form layer
        fc_layer = tf.nn.bias_add(tf.matmul(tf_input, W), b)

        return fc_layer

    def lossFunc(self):
        margin = 5.0
        with tf.variable_scope('loss_function') as scope:
            label = self.y
            # Euclidean distance squared
            euc_distance2 = tf.reduce_sum(
                tf.pow(tf.subtract(self.output1, self.output2), 2, name="euc_distance2"),
                1)
            # Euclidean distance
            # Add a small value 1e-6 to avoid NaN,
            # improving the stability of gradients calculation for sqrt
            # https://github.com/tensorflow/tensorflow/issues/4914
            euc_distance = tf.sqrt(euc_distance2 + 1e-6, name="euc_distance2")
            # Create loss
            pos = tf.multiply(label, euc_distance2, name="loss_positive")
            neg = tf.multiply(
                tf.subtract(1.0, label),
                tf.pow(tf.maximum(tf.subtract(margin, euc_distance), 0), 2),
                name="loss_negative")
            loss = tf.reduce_mean(tf.add(pos, neg), name="loss_function")
        
        return loss

    # Initialize optimizer
    def initOptimizer(self):
        # AdamOptimizer and GradientDescentOptimizer has different effect on the final results
        # GradientDescentOptimizer is probably better than AdamOptimizer in Siamese Network
        #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer