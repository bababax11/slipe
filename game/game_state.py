from typing import *
from enum import Enum, IntEnum, auto
import random
import numpy as np
from .errors import ChoiceOfMovementError, GameError


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
        self.board = np.array([
            [-1, -1, -2, -1, -1],
            [0] * 5,
            [0] * 5,
            [0] * 5,
            [1, 1, 2, 1, 1]
        ], dtype=np.int8)
        self.turn = 1  # +が先攻

    def to_input(self) -> np.ndarray:
        """強化学習用の入力"""
        arr = np.empty((4, 5, 5), dtype=bool)
        arr[0] = self.board == 1
        arr[1] = self.board == -1
        arr[2] = self.board == 2
        arr[3] = self.board == -2
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
            raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        direction = DIRECTIONS[drc]
        return self._move(i, j, direction)

    def _move(self, i: int, j: int, direction: np.ndarray) -> Winner:
        ij = np.array([i, j]) + direction
        while self.boundary_check(ij) and self.board[ij[0], ij[1]] == 0:
            ij += direction
        ij -= direction
        if tuple(ij) == (i, j):
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
            state = self.prior_checkmate()
            if state is not None:
                print('priority')
                return state

        while True:
            i = random.randint(0, 4)
            j = random.randint(0, 4)
            drc = random.randint(0, 3)
            try:
                state = self.move_by_drc(i, j, drc)
            except ChoiceOfMovementError:
                continue
            else:
                return state

    def prior_checkmate(self) -> Optional[Winner]:
        king = np.where(self.board == self.turn * 2)
        king = np.array([king[0][0], king[1][0]])
        if king[0] == 2:
            if king[1] > 2 and self.board[2, 1] != 0:
                return self._move(king[0], king[1], np.array([0, -1]))
            elif self.board[2, 3] != 0:
                return self._move(king[0], king[1], np.array([0, 1]))
        elif king[1] == 2:
            if king[0] > 2 and self.board[1, 2] != 0:
                return self._move(king[0], king[1], np.array([-1, 0]))
            elif self.board[3, 2] != 0:
                return self._move(king[0], king[1], np.array([1, 0]))
        return None
