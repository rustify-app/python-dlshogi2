import unittest

from cshogi import BLACK, BLACK_WIN, WHITE, WHITE_WIN

from pydlshogi2.features import make_eval_winrate, make_value_target


class ValueTargetTest(unittest.TestCase):
    """HCPE evaluations are stored from the side to move's perspective."""

    def test_positive_eval_favours_side_to_move_for_both_colours(self):
        self.assertGreater(make_eval_winrate(600, BLACK), 0.7)
        self.assertGreater(make_eval_winrate(600, WHITE), 0.7)
        self.assertLess(make_eval_winrate(-600, WHITE), 0.3)

    def test_blend_agrees_with_outcome_when_white_wins(self):
        # White to move, white is winning by +600 and wins the game.
        target = make_value_target(WHITE_WIN, 600, WHITE, val_lambda=0.5)
        self.assertGreater(target, 0.85)

    def test_blend_agrees_with_outcome_when_black_wins(self):
        target = make_value_target(BLACK_WIN, 600, BLACK, val_lambda=0.5)
        self.assertGreater(target, 0.85)


if __name__ == '__main__':
    unittest.main()
