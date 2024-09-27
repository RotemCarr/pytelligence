from utils.activation_functions import *
import numpy as np


class Neuron:

    def __init__(self, activation: float, weight: float, bias: float):
        self.activation = activation
        self.weight = weight
        self.bias = bias
        self.connections = None

    def __repr__(self):
        return f"Neuron(activation: {self.activation}, weight: {self.weight}, bias: {self.bias})"

    def activate(self, prev_layer):
        activation_vector = prev_layer.activation_vector
        weight_matrix = prev_layer.weight_matrix
        self.activation = sigmoid(np.matmul(weight_matrix, activation_vector) + prev_layer.bias_vector)

    @property
    def weight_vector(self):
        if self.connections is not None:
            return np.array([weight for weight in self.connections])

        return None


class InputNeuron(Neuron):

    def activate(self, _):
        return NotImplemented

    def __repr__(self):
        return f"Input(activation: {self.activation}, weight: {self.weight})"

    def __init__(self, input_activation, weight):
        super().__init__(activation=input_activation, weight=weight, bias=0)

