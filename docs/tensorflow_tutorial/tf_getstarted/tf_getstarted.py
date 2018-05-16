"""
Implementation of linear regression model using TensorFlow

Mathematical model for prediction: y = W*x + b
Loss function for training: sum(square(y - y_))
where:
    x: input
    W: weight(s), model parameter
    b: bias, model parameter
    y: predicted outcome
    y_: actual outcome (label)
"""

import tensorflow as tf

# model parameters
W = tf.Variable([0], dtype=tf.float32)
b = tf.Variable([0], dtype=tf.float32)

# model input and output
x = tf.placeholder(tf.float32, shape=None)  # if shape=None, then we can feed tensor with any shape
y = W*x + b
y_ = tf.placeholder(tf.float32)  # shape=None by default

# loss function
loss = tf.reduce_sum(tf.square(y - y_))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1., 2., 3., 4.]
y_train = [0., -1., -2., -3.]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in xrange(1000):
    sess.run(train, {x: x_train, y_: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y_: y_train})
print "W: %s, b: %s, loss: %s" % (curr_W, curr_b, curr_loss)

sess.close()
