"""
Model:    y = x*W + b
       loss = cross_entropy(y_, softmax(y))
where:
    x:  784-dim vector converted from 28x28 MNIST image
    W:  weights, tensor with shape [784, 10]
    b:  biases, 10-dim vector
    y_: labels, the 'correct' answers for the input images
"""

import tensorflow as tf

# Download MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Placeholders
x = tf.placeholder(tf.float32, [None, 784])     # 28x28 input images are flattened into 784-dim vectors
y_ = tf.placeholder(tf.float32, [None, 10])     # 10-dim one-hot vectors, since there are 10 possible outcomes (0-9)

# Model parameters
W = tf.Variable(tf.zeros([784, 10]))    # weights
b = tf.Variable(tf.zeros([10]))         # biases

# Model
y = tf.matmul(x, W) + b

# Softmax + Cross Entropy -> loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Training
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for _ in xrange(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluating
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
