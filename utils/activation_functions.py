import numpy as np


def relu(x: float):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def arctan(x):
    return np.arctan(x)


def tanh(x):
    return np.tanh(x)
