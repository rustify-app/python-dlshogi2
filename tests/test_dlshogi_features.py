import os
import random
import tempfile
import unittest

import numpy as np
import torch
from cshogi import Board, HuffmanCodedPosAndEval, move16
import cshogi.dlshogi as dlshogi

from pydlshogi2.dataloader import HcpeDataLoader
from pydlshogi2.features import feature_channels, input_feature_function
from pydlshogi2.network.policy_value_resnet import build_network


def random_hcpe(path, count, seed=0):
    rng = random.Random(seed)
    records = np.zeros(count, HuffmanCodedPosAndEval)
    moves = []
    board = Board()
    for record in records:
        if board.is_game_over() or board.move_number > 120:
            board.reset()
        board.to_hcp(record['hcp'])
        move = rng.choice(list(board.legal_moves))
        record['bestMove16'] = move16(move)
        moves.append((move, board.turn))
        board.push(move)
    records.tofile(path)
    return moves


class DlshogiFeatureTest(unittest.TestCase):
    def test_channel_count_matches_dlshogi(self):
        self.assertEqual(feature_channels('dlshogi'),
                         dlshogi.FEATURES1_NUM + dlshogi.FEATURES2_NUM)

    def test_features_match_cshogi_dlshogi(self):
        board = Board('lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2')
        out = np.ones((feature_channels('dlshogi'), 9, 9), dtype=np.float32)
        input_feature_function('dlshogi')(board, out)
        f1 = np.zeros((dlshogi.FEATURES1_NUM, 9, 9), dtype=np.float32)
        f2 = np.zeros((dlshogi.FEATURES2_NUM, 9, 9), dtype=np.float32)
        dlshogi.make_input_features(board, f1, f2)
        np.testing.assert_array_equal(out, np.concatenate((f1, f2)))

    def test_loader_uses_dlshogi_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'data.hcpe')
            moves = random_hcpe(path, 64)
            loader = HcpeDataLoader(path, 64, torch.device('cpu'), feature_set='dlshogi')
            x, labels, _ = loader.mini_batch(loader.data)
        self.assertEqual(tuple(x.shape), (64, feature_channels('dlshogi'), 9, 9))
        expected = [dlshogi.make_move_label(move, color) for move, color in moves]
        self.assertEqual(labels.tolist(), expected)

    def test_network_accepts_dlshogi_planes(self):
        model = build_network({'blocks': 1, 'channels': 16, 'fcl': 16, 'se': False,
                               'features': 'dlshogi'})
        policy, value = model(torch.zeros((2, feature_channels('dlshogi'), 9, 9)))
        self.assertEqual(tuple(policy.shape), (2, 27 * 81))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_mirror_augmentation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'data.hcpe')
            random_hcpe(path, 8)
            with self.assertRaises(ValueError):
                HcpeDataLoader(path, 8, torch.device('cpu'), feature_set='dlshogi',
                               mirror_augmentation=True)


if __name__ == '__main__':
    unittest.main()
