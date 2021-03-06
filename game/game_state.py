from typing import *
from enum import Enum, IntEnum, auto
import copy
import random
import numpy as np
from .errors import ChoiceOfMovementError


class Drc(IntEnum):
    """先手から見た方向を表す"""
    r = 0
    l = 1
    f = 2
    b = 3


DIRECTIONS = list(map(np.array, ([0, 1], [0, -1],
                                 [1, 0], [-1, 0])))


class Winner(Enum):
    not_ended = auto()
    plus = auto()
    minus = auto()


class GameState:

    def __init__(self) -> None:
        self.board = np.array(
            [[-1, -1, -2, -1, -1],
             [0,  0,  0,  0,  0],
             [0,  0,  0,  0,  0],
             [0,  0,  0,  0,  0],
             [1,  1,  2,  1,  1]], dtype=np.int8)
        self.turn = 1  # +が先攻

    def to_inputs(self, flip=False) -> np.ndarray:
        """強化学習用の入力"""
        arr = np.empty((1, 4, 5, 5), dtype=bool)
        if not flip:
            b = self.board
        else:
            b = np.flip(self.board * -1, 0)
        arr[0, 0] = b == 1
        arr[0, 1] = b == -1
        arr[0, 2] = b == 2
        arr[0, 3] = b == -2
        return arr

    def __repr__(self) -> str:
        return str(self.board)

    @staticmethod
    def boundary_check(ij: Union[Sequence[int], np.ndarray]) -> bool:
        return 0 <= ij[0] <= 4 and 0 <= ij[1] <= 4

    def move_d_normalize(self, i: int, j: int, d: np.ndarray) -> Winner:
        """規格化されてないd方向への移動.
        returnは勝利判定.
        無効な移動あるいはdが斜め移動ならChoiceOfMovementErrorを送出"""
        if d[0] != 0 and d[1] != 0:
            raise ChoiceOfMovementError(f"あらぬ方向{d}")
        d //= int(np.linalg.norm(d, np.inf))
        return self._move(i, j, d)

    def move_by_drc(self, i: int, j: int, drc: Drc) -> Winner:
        """DIRECTIONS[drc]方向への移動.
        returnは勝利判定.
        無効な移動ならChoiceOfMovementErrorを送出"""
        if self.board[i, j] * self.turn <= 0:
            raise ChoiceOfMovementError(f"選択したコマが色違いか存在しない {i, j}")
        direction = DIRECTIONS[drc]
        return self._move(i, j, direction)

    def _move(self, i: int, j: int, direction: np.ndarray) -> Winner:
        ij = np.array([i, j]) + direction
        while self.boundary_check(ij) and self.board[ij[0], ij[1]] == 0:
            ij += direction
        ij -= direction
        if ij[0] == i and ij[1] == j:
            raise ChoiceOfMovementError(f"移動できない方向{i, j}")
        self.board[ij[0], ij[1]], self.board[i, j] = \
            self.board[i, j], self.board[ij[0], ij[1]]

        return self._turn_change()

    def _turn_change(self) -> Winner:
        """勝利判定とターン交代"""
        center = self.board[2, 2]
        if center == 2:
            return Winner.plus  # 先手勝利
        elif center == -2:
            return Winner.minus  # 後手勝利
        self.turn *= -1
        return Winner.not_ended

    def random_play(self, decided_pb=1) -> Winner:
        """ランダムに手を打つ.
        ただし次の手で勝てるときはdecided_pbの確率で勝利手を打つ
        returnは勝利判定"""
        if random.random() < decided_pb:
            state_and_action = self.prior_checkmate()
            if state_and_action is not None:
                # print('priority')
                return state_and_action

        while True:
            i = random.randint(0, 4)
            j = random.randint(0, 4)
            drc = random.randint(0, 3)
            try:
                state = self.move_by_drc(i, j, drc)
            except ChoiceOfMovementError:
                continue
            else:
                return state, self.to_outputs_index(i, j, drc)

    def prior_checkmate(self) -> Optional[Tuple[Winner, int]]:
        king = np.where(self.board == self.turn * 2)
        king = np.array([king[0][0], king[1][0]])
        try:
            if king[0] == 2:
                if king[1] > 2 and self.board[2, 1] != 0:
                    return (self._move(king[0], king[1], np.array([0, -1])),
                            self.to_outputs_index(king[0], king[1], Drc.l))
                elif self.board[2, 3] != 0:
                    return (self._move(king[0], king[1], np.array([0, 1])),
                            self.to_outputs_index(king[0], king[1], Drc.r))
            elif king[1] == 2:
                if king[0] > 2 and self.board[1, 2] != 0:
                    return (self._move(king[0], king[1], np.array([-1, 0])),
                            self.to_outputs_index(king[0], king[1], Drc.b))
                elif self.board[3, 2] != 0:
                    return (self._move(king[0], king[1], np.array([1, 0])),
                            self.to_outputs_index(king[0], king[1], Drc.f))
        except ChoiceOfMovementError:
            pass
        return None

    @staticmethod
    def to_outputs_index(i: int, j: int, drc: Drc) -> int:
        return i * 20 + j * 4 + drc


    def outputs_to_move_max(self, outputs: 'array_like') -> Tuple[Winner, int]:
        """出力から最も高い確率のものに有効手を指す.
        returnは勝利判定と打った手"""
        outputs_ = outputs
        # outputs_ = copy.deepcopy(outputs)
        for _ in range(10):
            argmax = np.argmax(outputs_)
            outputs_[argmax] = -1.0
            try:
                state = self.move_by_drc(*np.unravel_index(argmax, (5, 5, 4)))
            except ChoiceOfMovementError:
                continue
            else:
                # print(argmax)
                # print(np.unravel_index(argmax, (5, 5, 4)))
                return state, argmax
        return self.random_play(0)

    def outputs_to_move_random(self, outputs: np.ndarray) -> Tuple[Winner, int]:
        """出力からランダムに有効手を指す.
        ただしoutputは確率分布になっている必要がある(1への規格化が必要).
        returnは勝利判定と打った手"""
        num_choices = min(np.sum(outputs != 0), 10)
        random_choices = np.random.choice(
            100, p=outputs, size=num_choices, replace=False)
        for r in random_choices:
            try:
                state = self.move_by_drc(*np.unravel_index(r, (5, 5, 4)))
            except ChoiceOfMovementError:
                continue
            else:
                # print(r)
                # print(np.unravel_index(r, (5, 5, 4)))
                return state, r

        return self.random_play(0)
