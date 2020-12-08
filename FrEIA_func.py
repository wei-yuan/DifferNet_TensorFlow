"""
Tutorial:
Making new layers and models via subclassing
(https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
"""
from math import exp
import tensorflow as tf
from tensorflow import keras
import numpy as np


class DummyData:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims


class Linear(keras.layers.Layer):
    """A single layer of fully connected layer"""
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class CustomizedFullyConnectedLayer(keras.layers.Layer):
    """
    the counterpart of class "F_fully_connected" in tensorflow of project differnet

    Fully connected transformation. In paper of DifferNet, this is a subnet called s and t.
    For more information, please see section 3.1.1 Architecture
    """
    def __init__(self):
        """State: weights and biases"""
        pass

    def call(self, inputs, **kwargs):
        """transformation"""
        pass