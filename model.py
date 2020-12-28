import tensorflow as tf
from tensorflow import keras
import numpy as np

import config as c
from FrEIA_func import PermuteLayer, GlowCouplingLayer, CustomizedFullyConnectedLayer #""",ReversibleGraphNet, OutputNode, InputNode, Node"""

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head(input_dim=c.n_feat):
    pass


class DifferNet:
    def __init__(self):
        self.feature_extractor = None  # alexnet with pretrained model
        self.nf = None

    def call(self, x):
        raise NotImplementedError()
