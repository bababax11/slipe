import os
from .configbase import ConfigBase


class Config(ConfigBase):

    def __init__(self):
        self.model = ModelConfig()
        self.Qlearn = QLearnConfig()


class ModelConfig(ConfigBase):

    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 0
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.learning_rate = 0.01

class QLearnConfig(ConfigBase):
    
    def __init__(self):
        self.DQN_MODE = False  # TrueがDQN、FalseがDDQN

        self.num_episodes = 60  # 総試行回数
        self.max_number_of_steps = 25  # 1試行のstep数
        self.goal_average_reward = 50  # この報酬を超えると学習終了
        self.num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
        self.gamma = 0.99    # 割引係数
        # ---
        self.hidden_size = 16               # Q-networkの隠れ層のニューロンの数
        self.learning_rate = 0.00001         # Q-networkの学習係数
        self.memory_size = 10000            # バッファーメモリの大きさ
        self.batch_size = 32                # Q-networkを更新するバッチの大記載
        # ---
        self.reward_win = 1
        self.reward_lose = -1
        self.ignore_auto = False
