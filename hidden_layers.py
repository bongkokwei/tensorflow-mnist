import tensorflow as tf
import numpy as np

def init_parameters():
    # Initializes parameters to build a neural network with tensorflow. The shapes are:
    # W1 : [784, 25]
    # b1 : [1, 25]
    # W2 : [25, 12]
    # b2 : [1, 12]
    # W3 : [12, 10]
    # b3 : [1, 10]

    W1 = tf.get_variable("W1", shape = [784, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", shape = [1, 25], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", shape = [25, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", shape = [1, 25], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", shape = [25, 10], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", shape = [1, 10], initializer = tf.zeros_initializer())

    # Python graphs are defined by their curly brackets, the keys are used to
    # identitfy an array
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters

def forward_propagation(X, parameters,keep_prob = 1):
    """
    Implements the forward propagation for the model:
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Retrieve the parameters from the dictionary "parameters"
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(X, W1), b1)
    A1 = tf.nn.dropout(tf.nn.relu(Z1), keep_prob=keep_prob)
    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.nn.dropout(tf.nn.relu(Z2), keep_prob=keep_prob)
    Z3 = tf.add(tf.matmul(A2, W3), b3)

    return Z3

def compute_cost(Y, Z3, parameters, m, lambd = 0):
    """
    Compute cost function with softmax cross entrophy and L2 regularization
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    L2 = (lambd/(2 * m)) * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))

    j = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = Z3)
    J = tf.reduce_mean(j) + L2

    return J
