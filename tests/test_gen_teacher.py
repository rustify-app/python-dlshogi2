import os
import random
import sys
import types
import unittest

from cshogi import BLACK, BLACK_WIN, WHITE, WHITE_WIN, Board, move16

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

import gen_teacher  # noqa: E402


class FakeEngine:
    """Plays the first legal move and reports scripted scores."""

    def __init__(self, scores, final='move'):
        self.scores = list(scores)
        self.final = final

    def new_game(self):
        pass

    def best(self, board):
        if not self.scores:
            return 0, self.final
        from cshogi import move_to_usi
        return self.scores.pop(0), move_to_usi(next(iter(board.legal_moves)))


def options(**overrides):
    values = dict(random_moves=0, max_plies=320, resign_score=3000)
    values.update(overrides)
    return types.SimpleNamespace(**values)


class GenTeacherTest(unittest.TestCase):
    def test_result_for_side_to_move(self):
        self.assertEqual(gen_teacher.result_for(BLACK, True), BLACK_WIN)
        self.assertEqual(gen_teacher.result_for(WHITE, True), WHITE_WIN)
        self.assertEqual(gen_teacher.result_for(BLACK, False), WHITE_WIN)

    def test_score_adjudication_labels_winner(self):
        # Black +100, White -200, Black +3500 -> Black wins at the third ply.
        engine = FakeEngine([100, -200, 3500])
        records, result = gen_teacher.play_game(engine, Board(), random.Random(0), options())
        self.assertEqual(result, BLACK_WIN)
        self.assertEqual([int(r['eval']) for r in records], [100, -200, 3500])

    def test_resign_labels_side_to_move_as_loser(self):
        engine = FakeEngine([10, 20], final='resign')
        records, result = gen_teacher.play_game(engine, Board(), random.Random(0), options())
        # Two moves played, Black to move resigns.
        self.assertEqual(len(records), 2)
        self.assertEqual(result, WHITE_WIN)

    def test_records_store_played_move(self):
        board = Board()
        first = next(iter(board.legal_moves))
        engine = FakeEngine([0], final='resign')
        records, _ = gen_teacher.play_game(engine, Board(), random.Random(0), options())
        self.assertEqual(int(records[0]['bestMove16']), move16(first))


if __name__ == '__main__':
    unittest.main()
