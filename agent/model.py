from typing import *
import time
import datetime
from collections import deque
import hashlib
import json
import os
from logging import getLogger
import numpy as np
from tqdm import tqdm, trange
# noinspection PyPep8Naming
import keras.backend as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.regularizers import l2

from .config import Config
from game.game_state import GameState, Winner


logger = getLogger(__name__)


class QNetwork:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.digest = None
        self.build()

    def build(self):
        mc = self.config.model
        in_x = x = Input((4, 5, 5))

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first",
                   kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        out = Dense(100, kernel_regularizer=l2(mc.l2_reg),
                    activation="softmax", name="out")(x)

        # x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
        #          activation="relu")(x)
        # value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
        #                   activation="tanh", name="value_out")(x)

        self.model = Model(in_x, out, name="slipe_model")
        self.model.compile(loss='mse', optimizer=Adam(lr=mc.learning_rate))
        self.model.summary()

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN) -> None:
        inputs = np.zeros((batch_size, 4, 5, 5))
        targets = np.zeros((batch_size, 100))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i] = state_b  # shape=(4, 5, 5)
            target = reward_b  # type: int

            # if not (next_state_b == 0).all():
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            retmainQs = self.model.predict(next_state_b)
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + gamma * \
                targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)[0][0]   # Qネットワークの出力
            # 教師信号 action_b: int <= 100
            targets[i, action_b] = target
            # epochsは訓練データの反復回数、verbose=0は表示なしの設定
            self.model.fit(inputs, targets, epochs=1, verbose=0)

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path) -> bool:
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(
                f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path) -> None:
        logger.debug(f"save model to {config_path}")
        # with open(config_path, "wt") as f:
        #     json.dump(self.model.get_config(), f)
        self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000) -> None:
        self.buffer = deque(maxlen=max_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(
            np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# [C]ｔ＋１での行動を返す
def get_action(board: np.ndarray, episode: int, mainQN: QNetwork, gs: GameState) -> Tuple[Winner, int]:
    # 徐々に最適行動のみをとる、ε-greedy法
    epsilon = 0.001 + 0.9 / (1.0+episode)

    if epsilon <= np.random.uniform(0, 1):
        retTargetQs = mainQN.model.predict(board)[0]
        s = gs.outputs_to_move_max(retTargetQs)  # 最大の報酬を返す行動を選択する

    else:
        s = gs.random_play()  # ランダムに行動する

    return s


if __name__ == '__main__':
    DQN_MODE = True  # 1がDQN、0がDDQNです
    LENDER_MODE = False  # 0は学習後も描画なし、1は学習終了後に描画する

    num_episodes = 60  # 総試行回数
    max_number_of_steps = 25  # 1試行のstep数
    goal_average_reward = 50  # この報酬を超えると学習終了
    num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
    total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
    gamma = 0.99    # 割引係数
    islearned = False  # 学習が終わったフラグ
    isrender = False  # 描画フラグ
    # ---
    hidden_size = 16               # Q-networkの隠れ層のニューロンの数
    learning_rate = 0.00001         # Q-networkの学習係数
    memory_size = 10000            # バッファーメモリの大きさ
    batch_size = 32                # Q-networkを更新するバッチの大記載

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(config=Config())     # メインのQネットワーク
    targetQN = QNetwork(config=Config())   # 価値を計算するQネットワーク
    # plot_model(mainQN.model, to_file='QNetwork.png', show_shapes=True)        # Qネットワークの可視化
    memory = Memory(max_size=memory_size)

    for episode in trange(num_episodes):  # 試行数分繰り返す
        gs = GameState()
        state = gs.random_play()  # 1step目は適当な行動をとる
        episode_reward = 0

        targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする

        for t in range(max_number_of_steps + 1):  # 1試行のループ
            board = gs.to_inputs()
            # if islearned and LENDER_MODE:  # 学習終了したらcartPoleを描画する
            #     env.render()
            #     print(state[0, 0])  # カートのx位置を出力するならコメントはずす

            state, action = get_action(
                board, episode, mainQN, gs)   # 時刻tでの行動を決定する
            # next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
            # next_state = np.reshape(next_state, [1, 4])     # list型のstateを、1行4列の行列に変換
            
            # verbose ==========
            # if t % 10 == 9:
            #     print(gs)
            # ==================

            # 報酬を設定し、与える
            # next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
            if state == Winner.minus:
                reward = 1  # 報酬クリッピング、報酬は1, 0, -1に固定
            else:
                reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）

            next_board = gs.to_inputs()

            # board = next_board  # 状態更新

            # Qネットワークの重みを学習・更新する replay
            if (len(memory) > batch_size) and not islearned:
                mainQN.replay(memory, batch_size, gamma, targetQN)

            if DQN_MODE:
                targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする

            # 1施行終了時の処理
            if state != Winner.not_ended:
                episode_reward += reward  # 合計報酬を更新
                memory.add((board, action, reward, next_board))     # メモリの更新する
                total_reward_vec = np.hstack(
                    (total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d/%d: Episode finished after %d time steps / mean %f winner: %s'
                      % (episode+1, num_episodes, t + 1, total_reward_vec.mean(),
                      'plus' if state == Winner.plus else 'minus'))
                break

            state, _ = gs.random_play()

            if state == Winner.plus:
                reward = -1  # 立ったまま195step超えて終了時は報酬
            else:
                reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）
            
            episode_reward += reward  # 合計報酬を更新
            memory.add((board, action, reward, next_board))     # メモリの更新する


            # 1施行終了時の処理
            if state != Winner.not_ended:
                total_reward_vec = np.hstack(
                    (total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d/%d: Episode finished after %d time steps / mean %f winner: %s'
                      % (episode+1, num_episodes, t + 1, total_reward_vec.mean(),
                      'plus' if state == Winner.plus else 'minus'))
                break

        # 複数施行の平均報酬で終了を判断
        if total_reward_vec.mean() >= goal_average_reward:
            print('Episode %d train agent successfuly!' % episode)
            islearned = True
            if isrender == False:   # 学習済みフラグを更新
                isrender = True
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    mainQN.save(None, f"results/001_QLearning/{d}-mainQN-60times.h5")
    targetQN.save(None, f"results/001_QLearning/{d}-targetQN-60times.h5")
