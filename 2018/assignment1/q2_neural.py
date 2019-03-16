#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    h = X.dot(W1) + b1 # (M, Dx) * (Dx, H) -> (M, H)
    sig_h = sigmoid(h)
    y = sig_h.dot(W2) + b2 # (M, H) * (H, Dy) -> (M, Dy)
    softmax_y = softmax(y)
    cost = -np.sum(labels * np.log(softmax_y))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    d_y = softmax_y - labels # (M, Dy) https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    d_W2 = sig_h.T.dot(d_y) # (H, M) * (M, Dy) -> (H, Dy)
    d_b2 = np.sum(d_y, axis=0, keepdims=True) # (M, Dy) -> (, Dy)

    d_sig_h = d_y.dot(W2.T) # (M, Dy) * (Dy, H) -> (M, H)
    d_h = sigmoid_grad(sig_h) * d_sig_h # (M, H)
    d_W1 = X.T.dot(d_h) # (Dx, M) * (M, H) = (Dx, H)
    d_b1 = np.sum(d_h, axis=0, keepdims=True) # (M, H) -> (, H)
    gradW1, gradb1, gradW2, gradb2 = d_W1, d_b1, d_W2, d_b2
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    N = 20
    dimensions = [100, 30, 10]
    data = np.random.standard_normal((N, dimensions[0]))
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
            dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
