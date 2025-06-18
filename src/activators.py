import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - x**2
