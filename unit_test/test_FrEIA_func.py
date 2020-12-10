import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn


class TestBasicLayers:

    def test_torch_tensor(self):
        assert (torch.ones((2, 1)).numpy() == np.ones((2, 1))).all()

    def test_tf_tensor(self):
        assert (tf.ones((2, 1)).numpy() == np.ones((2, 1))).all()

    def test_dropout(self):
        """
        torch.nn.Dropout(p: float = 0.5, inplace: bool = False)
        p stands for [p]robability
        see https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

        tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
        see https://keras.io/api/layers/regularization_layers/dropout/

        how to check empty numpy array?
        see How can I check whether a numpy array is empty or not? (https://stackoverflow.com/a/56715195/1115215)
        """
        x = torch.ones((2, 1))
        torch_dropout_layer = nn.Dropout(p=0.0)
        torch_y = torch_dropout_layer(x)

        x = tf.ones((2, 1))
        tf_dropout_layer = keras.layers.Dropout(rate=0.0)
        # Note that the Dropout layer only applies when training is set to True
        # such that no values are dropped during inference.
        tf_y = tf_dropout_layer(x, training=True)
        assert torch_y.numpy().shape, "array torch_y is empty"
        assert tf_y.numpy().shape, "array tf_y is empty"
        assert (np.array_equal(torch_y.numpy(), tf_y.numpy()))

    def test_linear(self):
        """
        torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        tf.keras.layers.Dense(
            units,
            activation=None, use_bias=True,
            kernel_initializer="glorot_uniform", bias_initializer="zeros",
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
            **kwargs
        )
        see https://keras.io/api/layers/core_layers/dense/
        """
        x = torch.ones((2, 1))
        with torch.no_grad():
            torch_linear_layer = nn.Linear(in_features=1, out_features=2)  # mat size: (in_features x out_features)
            torch_linear_layer.weight.copy_(torch.ones(2, 1))
            torch_linear_layer.bias.copy_(torch.zeros(2,))
            torch_y = torch_linear_layer(x)

        x = tf.ones((2, 1))
        init = tf.constant_initializer(value=1)
        tf_linear_layer = keras.layers.Dense(units=2, kernel_initializer=init)
        tf_y = tf_linear_layer(x)

        torch_np = torch_y.detach().numpy()
        tf_np = tf_y.numpy()
        assert torch_np.shape, "array torch_y is empty"
        assert tf_np.shape, "array tf_y is empty"
        assert (np.array_equal(torch_np, tf_np))


if __name__ == '__main__':
    pass
