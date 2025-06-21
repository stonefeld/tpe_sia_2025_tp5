import numpy as np


def sigmoid(x):
    # Numerically stable sigmoid function
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_prime(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - x**2


def get_activation_function(name):
    if name == "sigmoid":
        return sigmoid, sigmoid_prime
    elif name == "tanh":
        return tanh, tanh_prime
    else:
        raise ValueError(f"Unknown activation function: {name}")
