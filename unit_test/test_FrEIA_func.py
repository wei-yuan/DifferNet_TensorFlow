import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
import pytest
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn


class TestBasicLayers:
    def test_basic(self):
        assert True

    def test_dropout(self):
        x = torch.ones((2, 1))
        torch_dropout = nn.Dropout(p=0.0)
        torch_y = torch_dropout(x)

        x = tf.ones((2, 1))
        tf_dropout_layer = keras.layers.Dropout(rate=0.0)
        tf_y = tf_dropout_layer(x)
        assert torch_y == tf_y


if __name__ == '__main__':
    unittest.main()
