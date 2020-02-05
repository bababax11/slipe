from typing import *
from enum import IntEnum
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

    def __repr__(self):
        return str(self.board)

    @staticmethod
    def boundary_check(ij: Union[Sequence[int], np.ndarray]) -> bool:
        return 0 <= ij[0] <= 4 and 0 <= ij[1] <= 4

    def move_d_normalize(self, i: int, j: int, d: np.ndarray) -> int:
        if d[0] != 0 and d[1] != 0:
            raise ChoiceOfMovementError(f"あらぬ方向{d}")
        d //= int(np.linalg.norm(d, np.inf))
        return self.move(i, j, d)

    def move_by_drc(self, i: int, j: int, drc: Drc) -> int:
        if self.board[i, j] * self.turn <= 0:
            raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        direction = DIRECTIONS[drc]
        return self.move(i, j, direction)

    def move(self, i: int, j: int, direction: np.ndarray) -> int:
        ij = np.array([i, j]) + direction
        while self.boundary_check(ij) and self.board[ij[0], ij[1]] == 0:
            ij += direction
        ij -= direction
        if tuple(ij) == (i, j):
            raise ChoiceOfMovementError(f"移動できない方向{i, j}")
        self.board[ij[0], ij[1]], self.board[i, j] = \
            self.board[i, j], self.board[ij[0], ij[1]]

        return self.turn_change()

    def turn_change(self) -> int:
        center = self.board[2, 2]
        if center == 2:
            return 1  # 先手勝利
        elif center == -2:
            return -1  # 後手勝利
        self.turn *= -1
        return 0

    def random_play(self) -> int:
        while True:
            i = random.randint(0, 4)
            j = random.randint(0, 4)
            drc = random.randint(0, 3)
            try:
                state = self.move_by_drc(i, j, drc)
            except GameError:
                continue
            else:
                return state
