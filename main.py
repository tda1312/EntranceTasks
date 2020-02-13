from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

# Modules
import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from siamese import Siamese

# Prepare data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# mnist = tensorflow_datasets.load('mnist')
# mnist = tf.keras.datasets.mnist.load_data()

# Instantiate Siamese network
siamese = Siamese()
# Initialize tensorflow session
saver = tf.train.Saver()
session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Check pre-trained model
if os.path.exists('model/'):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("Model found. Do you want to load and continue training [yes/no]?")
    if input_var == 'yes':
        saver.restore(session, 'model/siamese_model')

# Train model
for step in range(10000):
    x1, y1 = mnist.train.next_batch(128)
    x2, y2 = mnist.train.next_batch(128)
    y = (y1 == y2).astype('float')
    
    _, loss = session.run(
        [siamese.optimizer, siamese.loss],
        feed_dict = {
            siamese.x1: x1,
            siamese.x2: x2,
            siamese.y: y})

    if np.isnan(loss):
        print("Model unstable, diverged with loss = NaN")
        quit()

    # Keep track of training steps
    if step % 10 == 0:
        print("Step {}: loss {:.3}".format(step, loss))
    
    # Save model
    if step % 1000 == 0 and step > 0:
        if not os.path.exists('model/'):
            os.makedirs('model/')
        saver.save(session, 'model/siamese_model')
        # Test model
        embed = session.run(
            siamese.output1,
            feed_dict = {siamese.x1: mnist.test.images})
        embed.tofile('embed.txt')
