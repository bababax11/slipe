import unittest
import numpy as np
from game.game_state import GameState


class TestGameState(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()

    def test_output_to_move_max(self):
        output = np.linspace(0.0, 1.0, 100)
        self.gs.output_to_move_max(output)
        self.assertTrue((self.gs.board ==
                         np.array([[-1, -1, -2, -1, -1],
                                   [0,  0,  0,  0,  1],
                                   [0,  0,  0,  0,  0],
                                   [0,  0,  0,  0,  0],
                                   [1,  1,  2,  1,  0]])
                         ).all())
        self.gs.output_to_move_max(output)
        self.assertTrue((self.gs.board ==
                         np.array([[-1, -1, -2,  0, -1],
                                   [0,  0,  0,  0,  1],
                                   [0,  0,  0,  0,  0],
                                   [0,  0,  0, -1,  0],
                                   [1,  1,  2,  1,  0]])
                         ).all())

    def test_output_to_move_random(self):
        output = np.linspace(0.0, 1.0, 100)
        output /= np.sum(output)
        self.gs.output_to_move_random(output)
