"""Sanity check of FrEIA function
"""
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

    def test_torch_and_tf_equaivalence(self):
        """
        see https://johnwlambert.github.io/pytorch-tutorial/#data-types
        """
        max_val_of_int64 = 9223372036854775807
        arr = [[max_val_of_int64], [max_val_of_int64]]
        assert (
            np.array_equal(tf.constant(arr, dtype=tf.int64).numpy(), torch.LongTensor(arr).numpy())
        )

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
            # Initialize nn.Linear with specific weights
            # see https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/3
            torch_linear_layer.weight.copy_(torch.ones(2, 1))
            torch_linear_layer.bias.copy_(torch.zeros(2, ))
            torch_y = torch_linear_layer(x)

        x = tf.ones((2, 1))
        # Custom weight initialization tensorflow tf.layers.dense
        # see https://stackoverflow.com/questions/49501538/custom-weight-initialization-tensorflow-tf-layers-dense
        init = tf.constant_initializer(value=1)
        tf_linear_layer = keras.layers.Dense(units=2, kernel_initializer=init)
        tf_y = tf_linear_layer(x)

        torch_np = torch_y.detach().numpy()
        tf_np = tf_y.numpy()
        assert torch_np.shape, "array torch_y is empty"
        assert tf_np.shape, "array tf_y is empty"
        assert (np.array_equal(torch_np, tf_np))

    def test_relu(self):
        """
        torch.nn.ReLU(inplace: bool = False)
        see https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

        tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=0, **kwargs)
        see https://keras.io/api/layers/activation_layers/relu/
        """
        x1 = torch.ones((2, 1))
        torch_relu_layer = nn.ReLU()
        torch_y1 = torch_relu_layer(x1)

        x1 = tf.ones((2, 1))
        tf_relu_layer = keras.layers.ReLU()
        tf_y1 = tf_relu_layer(x1)

        torch_np = torch_y1.detach().numpy()
        tf_np = tf_y1.numpy()
        assert torch_np.shape, "array torch_y is empty"
        assert tf_np.shape, "array tf_y is empty"
        assert (np.array_equal(torch_np, tf_np))

        # x_minus_1 = np.full((2, 1), -1)
        # torch_minus_1 = torch_relu_layer(x_minus_1)

    # TODO: trace code and fix failed test
    def test_batch_norm(self):
        """
        2020/12/16 Need to trace code

        torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        see https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        source code https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py

        tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros", gamma_initializer="ones",
            moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None,
            beta_constraint=None, gamma_constraint=None,
            **kwargs
        )

        BatchNormalization layer
        see https://keras.io/api/layers/normalization_layers/batch_normalization/
        """
        epsilon = 10 ** -4
        momentum = 10 ** -1
        batch = 2
        x1 = torch.ones((batch, 1))  # (N, C)
        torch_batch_norm_layer = nn.BatchNorm1d(
            num_features=1,
            eps=epsilon,
            momentum=momentum
        )
        torch_y1 = torch_batch_norm_layer(x1)

        x1 = tf.ones((batch, 1))  # C
        tf_batch_norm_layer = keras.layers.BatchNormalization(
            epsilon=epsilon,
            momentum=momentum
        )
        tf_y1 = tf_batch_norm_layer(x1)

        torch_np = torch_y1.detach().numpy()
        tf_np = tf_y1.numpy()
        assert torch_np.shape, "array torch_np is empty"
        assert tf_np.shape, "array tf_np is empty"
        assert (np.array_equal(torch_np, tf_np))


class TestGlowCouplingLayer:
    """This unit test is to make sure every operation works like the API in PyTorch Library"""

    def test_tenor_narrow(self):
        """
        torch.narrow(input, dim, start, length)
        Returns a new tensor that is a narrowed version of input tensor.
        see https://pytorch.org/docs/stable/generated/torch.narrow.html

        TensorFlow can slices tensor in two means:
            1. Use the indexing operator (based on tf.slice())
            2. tf.gather()
        here, we choose the first one, which is more intuitive
        see https://stackoverflow.com/a/35158370/1115215
        """
        start = 0
        length = 2
        assert np.array_equal(
            torch.narrow(input=torch.ones((2, 1)), dim=0, start=start, length=length).numpy(),
            tf.ones((2, 1))[start:start + length, :].numpy()
        )


if __name__ == '__main__':
    pass
