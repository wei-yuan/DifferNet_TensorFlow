"""
Tutorial:
Making new layers and models via subclassing
(https://keras.io/guides/making_new_layers_and_models_via_subclassing/)

Convention:
    TensorFlow: [Batch, height, width, color-channel]
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

    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        """State: weights and biases"""

        super(CustomizedFullyConnectedLayer, self).__init__()
        if not internal_size:
            internal_size = 2 * size

        self.d1 = keras.layers.Dropout(rate=dropout)
        self.d2 = keras.layers.Dropout(rate=dropout)
        self.d2b = keras.layers.Dropout(rate=dropout)

        self.fc1 = keras.layers.Linear(size_in, internal_size)
        self.fc2 = keras.layers.Linear(internal_size, internal_size)
        self.fc2b = keras.layers.Linear(internal_size, internal_size)
        self.fc3 = keras.layers.Linear(internal_size, size)

        self.nl1 = keras.layers.ReLU()
        self.nl2 = keras.layers.ReLU()
        self.nl2b = keras.layers.ReLU()

    def call(self, x):
        """transformation"""
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.nl2(self.d2(self.fc2(out)))
        out = self.nl2b(self.d2b(self.fc2b(out)))
        out = self.fc3(out)
        return out


class PermuteLayer(keras.layers.Layer):
    def __init__(self, dims_in, seed):
        super(PermuteLayer, self).__init__()
        self.in_channels = dims_in[0][0]

        np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)
        np.random.seed()

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = tf.constant(self.perm, dtype=tf.int64)
        self.perm_inv = tf.constant(self.perm_inv, dtype=tf.int64)

    def call(self, x, rev=False):
        if not rev:
            return [x[0][:, self.perm]]
        else:
            return [x[0][:, self.perm_inv]]

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class GlowCouplingLayer(keras.layers.Layer):

    def __init__(self, dims_in, f_class=CustomizedFullyConnectedLayer, f_args={}, clamp=5.):
        super(GlowCouplingLayer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = f_class(self.split_len1, self.split_len2 * 2, **f_args)
        self.s2 = f_class(self.split_len2, self.split_len1 * 2, **f_args)

    def e(self, s):
        return tf.math.exp(self.log_e(s))

    def log_e(self, s):
        return self.clamp * 0.636 * tf.math.atan(s / self.clamp)

    def call(self, inputs, **kwargs):
        raise NotImplementedError()

    def jacobian(self):
        raise NotImplementedError()

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
