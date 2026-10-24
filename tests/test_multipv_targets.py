import os
import tempfile
import unittest

import numpy as np
import torch
from cshogi import Board, BLACK_WIN, HuffmanCodedPosAndEval, move16

from pydlshogi2.dataloader import HcpeDataLoader
from pydlshogi2.features import make_move_label
from pydlshogi2.losses import policy_cross_entropy
from utils.make_multipv_targets import UsiEngine, softmax


class SparsePolicyLossTest(unittest.TestCase):
    def test_one_hot_teacher_matches_hard_cross_entropy(self):
        logits = torch.tensor([[0.2, -0.1, 1.0], [1.2, 0.1, -0.5]])
        labels = torch.tensor([2, 0])
        teacher_labels = torch.tensor([[2, 0], [0, 0]])
        teacher_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        expected = torch.nn.functional.cross_entropy(logits, labels)
        actual = policy_cross_entropy(
            logits, labels, teacher_labels, teacher_probs)
        self.assertTrue(torch.allclose(actual, expected))

    def test_candidate_probabilities_weight_their_policy_logits(self):
        logits = torch.tensor([[0.0, 2.0, -1.0]])
        hard_label = torch.tensor([0])
        target_labels = torch.tensor([[0, 1]])
        probabilities = torch.tensor([[0.75, 0.25]])
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        expected = -(log_probs[0, 0] * 0.75 + log_probs[0, 1] * 0.25)
        actual = policy_cross_entropy(
            logits, hard_label, target_labels, probabilities)
        self.assertTrue(torch.allclose(actual, expected))


class MultiPvTargetDataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.hcpe_path = os.path.join(self.temp_dir.name, 'sample.hcpe')
        self.target_path = os.path.join(self.temp_dir.name, 'targets.npz')
        self.records = np.zeros(1, dtype=HuffmanCodedPosAndEval)
        self.board = Board()
        self.board.to_hcp(self.records[0]['hcp'])
        moves = list(self.board.legal_moves)
        self.records[0]['bestMove16'] = move16(moves[0])
        self.records[0]['gameResult'] = BLACK_WIN
        self.records[0]['eval'] = 0
        self.records.tofile(self.hcpe_path)

    def write_targets(self, hcp, labels, probabilities):
        hcp_bytes = bytes(hcp)
        hcp_array = np.frombuffer(hcp_bytes, dtype=np.uint8).reshape(1, -1)
        np.savez_compressed(
            self.target_path,
            hcp=hcp_array,
            labels=np.asarray([labels], dtype=np.int64),
            probabilities=np.asarray([probabilities], dtype=np.float32))

    def test_loader_returns_teacher_distribution_for_matching_position(self):
        first, second = list(self.board.legal_moves)[:2]
        labels = [make_move_label(first, self.board.turn),
                  make_move_label(second, self.board.turn)]
        self.write_targets(self.records[0]['hcp'], labels, [0.7, 0.3])
        loader = HcpeDataLoader(
            self.hcpe_path, 1, torch.device('cpu'),
            teacher_targets=self.target_path)
        batch = loader.mini_batch(self.records)
        self.assertEqual(len(batch), 5)
        self.assertEqual(batch[3].tolist(), [labels])
        self.assertTrue(torch.allclose(
            batch[4], torch.tensor([[0.7, 0.3]]), atol=1e-6))
        loader.executor.shutdown(wait=True)

    def test_loader_without_teacher_targets_keeps_hard_label_format(self):
        loader = HcpeDataLoader(
            self.hcpe_path, 1, torch.device('cpu'))
        batch = loader.mini_batch(self.records)
        self.assertEqual(len(batch), 3)
        loader.executor.shutdown(wait=True)

    def test_unmatched_position_falls_back_to_record_move(self):
        other = Board()
        other.push(list(other.legal_moves)[0])
        other_hcp = np.zeros(1, dtype=HuffmanCodedPosAndEval)
        other.to_hcp(other_hcp[0]['hcp'])
        self.write_targets(other_hcp[0]['hcp'], [0, 1], [0.5, 0.5])
        loader = HcpeDataLoader(
            self.hcpe_path, 1, torch.device('cpu'),
            teacher_targets=self.target_path)
        batch = loader.mini_batch(self.records)
        self.assertEqual(batch[3][0, 0].item(), batch[1][0].item())
        self.assertEqual(batch[4].tolist(), [[1.0, 0.0]])
        loader.executor.shutdown(wait=True)

    def test_loader_rejects_invalid_probability_rows(self):
        hcp_array = np.frombuffer(
            bytes(self.records[0]['hcp']), dtype=np.uint8).reshape(1, -1)
        np.savez_compressed(
            self.target_path, hcp=hcp_array,
            labels=np.asarray([[0, 1]], dtype=np.int64),
            probabilities=np.asarray([[0.2, 0.2]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            HcpeDataLoader(
                self.hcpe_path, 1, torch.device('cpu'),
                teacher_targets=self.target_path)


class MultiPvParserTest(unittest.TestCase):
    def test_parses_centipawn_and_mate_scores(self):
        cp = UsiEngine._parse_info(
            'info depth 12 multipv 2 score cp 45 nodes 100 pv 2g2f 3c3d')
        mate = UsiEngine._parse_info(
            'info depth 12 multipv 1 score mate 3 nodes 100 pv 7g7f')
        self.assertEqual(cp, (2, 45, '2g2f'))
        self.assertEqual(mate, (1, 29997, '7g7f'))

    def test_softmax_prefers_higher_engine_scores(self):
        probs = softmax([100, 0, -100], 100)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
