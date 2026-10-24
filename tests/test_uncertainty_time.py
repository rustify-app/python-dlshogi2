import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pydlshogi2.player.mcts_player import MCTSPlayer, uncertainty_time_factor


class UncertaintyTimeTest(unittest.TestCase):
    def test_uniform_policy_receives_maximum_time(self):
        self.assertAlmostEqual(uncertainty_time_factor([0.25] * 4), 1.25)

    def test_certain_policy_receives_minimum_time(self):
        self.assertAlmostEqual(uncertainty_time_factor([1.0, 0.0, 0.0]), 0.75)

    def test_missing_or_invalid_policy_uses_neutral_time(self):
        self.assertEqual(uncertainty_time_factor(None), 1.0)
        self.assertEqual(uncertainty_time_factor([0.0, 0.0]), 1.0)
        self.assertEqual(uncertainty_time_factor([np.nan, 1.0]), 1.0)

    def test_zero_elapsed_milliseconds_does_not_divide_by_zero(self):
        player = MCTSPlayer.__new__(MCTSPlayer)
        player.halt = None
        player.tree = SimpleNamespace(current_head=SimpleNamespace(child_move=[1, 2]))
        player.root_board = SimpleNamespace(move_number=1)
        player.begin_time = 100.0
        player.time_limit = 0
        player.minimum_time = 0
        with patch('pydlshogi2.player.mcts_player.time.time', return_value=100.0):
            self.assertFalse(player.check_interruption())


if __name__ == '__main__':
    unittest.main()
