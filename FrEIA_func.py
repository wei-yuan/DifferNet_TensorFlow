from math import exp
import tensorflow as tf
import numpy as np


class DummyData:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims


class FullyConnectedLayer:
    def __init__(self):
        pass