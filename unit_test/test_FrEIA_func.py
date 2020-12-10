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
        """
        x = torch.ones((2, 1))
        torch_dropout = nn.Dropout(p=0.0)
        torch_y = torch_dropout(x)

        x = tf.ones((2, 1))
        tf_dropout_layer = keras.layers.Dropout(rate=0.0)
        # Note that the Dropout layer only applies when training is set to True
        # such that no values are dropped during inference.
        tf_y = tf_dropout_layer(x, training=True)
        assert (torch_y.numpy() == tf_y.numpy()).any()


if __name__ == '__main__':
    pass
