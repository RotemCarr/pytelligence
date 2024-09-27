from utils.activation_functions import *
import numpy as np


class Neuron:

    def __init__(self, weight: float, bias: float):
        self.activation = None
        self.weight = weight
        self.bias = bias
        self.connections = None

    def __repr__(self):
        return f"Neuron(activation: {self.activation}, weight: {self.weight}, bias: {self.bias})"

    # def activate(self, layer):
    #     activation_vector = layer.activation_vector
    #     weight_matrix = layer.weight_matrix
    #     self.activation = sigmoid(np.matmul(weight_matrix, activation_vector) + layer.bias_vector)

    @property
    def weight_vector(self):
        if self.connections is not None:
            return np.array([neuron.weight for neuron in self.connections])

        return None


class InputNeuron(Neuron):

    def __repr__(self):
        return f"Input(activation: {self.activation}, weight: {self.weight})"

    def __init__(self, input_activation, weight):
        super().__init__(weight=weight, bias=0)
        self.activation = input_activation
