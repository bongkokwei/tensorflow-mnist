import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from hidden_layers import *

# Hyperparameters tuning
num_epoch = 200
learning_rate = 0.1
minibatch_size = 512
l2 = 0.001
keep_prob = 1

# Applies exponential decay to the learning rate.
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)

# training and test data
# each training example is a 28x28 image file. We flatten the 2D array to a 1D
# array of 784 elements
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
m_train = 55000 # number taken from tutorial
m_test = 1000
m_validation = 5000
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10]) # training labels

# Initializes parameters
parameters = init_parameters()

# forward propagation
Z3 = forward_propagation(X, parameters, keep_prob = keep_prob)

# compute cost
J = compute_cost(Y, Z3, parameters, minibatch_size, lambd = l2)

# train NN with Adam Optimizer and exponential decay of learning rate
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(J, global_step=global_step)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(num_epoch):
    num_minibatches = int(m_train / minibatch_size)
    epoch_cost = 0.0 #reset after each epoch
    for _ in range(num_minibatches):
        minibatch_X, minibatch_Y = mnist.train.next_batch(minibatch_size)
        _, minibatch_J = sess.run([train_step, J], feed_dict={X: minibatch_X, Y: minibatch_Y})
        epoch_cost += minibatch_J / num_minibatches

    if epoch % 10 == 0:
        print("Cost after epoch %i: %f" % (epoch, epoch_cost))

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Z3,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Train Accuracy:", accuracy.eval({X: mnist.train.images, Y: mnist.train.labels}))
print("Test Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
