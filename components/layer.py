import numpy as np
from typing import List
from components.neuron import Neuron, InputNeuron
from utils.activation_functions import sigmoid


class Layer:

    def __str__(self):
        return f"{[str(neuron) for neuron in self]}"

    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons
        self.activation_vector = None
        self.weight_matrix = None
        self.bias_vector = None

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        current = self.current
        try:
            self.current += 1
            return self.neurons[current]

        except IndexError:
            raise StopIteration

    @type
    def _layer_type(self):
        return type(self)

    def _set_activation_vector(self):
        self.activation_vector = np.array([neuron.activation for neuron in self.neurons])

    def _set_bias_vector(self):
        self.bias_vector = np.array([neuron.bias for neuron in self.neurons])

    def _connect_neurons(self, next_layer: _layer_type):
        for neuron in next_layer:
            neuron.connections = self.neurons

    def _set_weight_matrix(self, next_layer: _layer_type):
        self.weight_matrix = np.array([neuron.weight_vector for neuron in next_layer.neurons])

    def _activate_layer(self, prev_layer: _layer_type):
        self.activation_vector = sigmoid(
            np.matmul(prev_layer.weight_matrix, prev_layer.activation_vector)
            + self.bias_vector)

    @staticmethod
    def create_input_layer(neurons: List[InputNeuron]):
        layer = Layer(neurons)
        layer._set_activation_vector()
        layer.bias_vector = np.array(len(neurons)*[0])
        return layer

    @staticmethod
    def create_hidden_layer(neurons: List[Neuron], prev_layer: _layer_type):
        layer = Layer(neurons)
        layer._set_activation_vector()
        layer._set_bias_vector()
        prev_layer._connect_neurons(layer)
        prev_layer._set_weight_matrix(layer)
        layer._activate_layer(prev_layer)

        return layer

    @staticmethod
    def create_output_layer(neurons: List[Neuron], prev_layer: _layer_type):
        return NotImplemented
