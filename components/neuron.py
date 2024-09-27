import random

from utils.activation_functions import *
import numpy as np


class Neuron:

    def __init__(self, bias: float):
        self.activation = None
        self.bias = bias
        self.connections = None

    def __repr__(self):
        return f"Neuron(activation: {self.activation}, bias: {self.bias})"

    @property
    def weight_vector(self):
        if self.connections is not None:
            return np.array([random.uniform(-1, 1) for _ in self.connections])

        return None


class InputNeuron(Neuron):

    def __repr__(self):
        return f"Input(activation: {self.activation})"

    def __init__(self, input_activation):
        super().__init__(bias=0)
        self.activation = input_activation
