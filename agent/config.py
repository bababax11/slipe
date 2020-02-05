import os
# from moke_config import ConfigBase


class Config:

    def __init__(self):
        self.model = ModelConfig()


class ModelConfig:

    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 0
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.learning_rate = 0.01
