import os
import tempfile
import unittest

import numpy as np
import torch
from cshogi import Board, BLACK_WIN, HuffmanCodedPosAndEval, move16

from pydlshogi2.dataloader import HcpeDataLoader


class SelfPlayDataSamplingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        boards = [Board()]
        for _ in range(3):
            board = Board()
            for _ in range(len(boards)):
                board.push(list(board.legal_moves)[0])
            boards.append(board)
        self.records = []
        for index, board in enumerate(boards):
            record = np.zeros(1, dtype=HuffmanCodedPosAndEval)
            board.to_hcp(record[0]['hcp'])
            record[0]['bestMove16'] = move16(list(board.legal_moves)[0])
            record[0]['gameResult'] = BLACK_WIN
            record[0]['eval'] = index
            self.records.append(record[0])

    def write_file(self, name, indices):
        path = os.path.join(self.temp_dir.name, name)
        np.asarray([self.records[i] for i in indices],
                   dtype=HuffmanCodedPosAndEval).tofile(path)
        return path

    def test_deduplication_keeps_the_newest_record_for_a_position(self):
        older = self.write_file('old.hcpe', [0, 1])
        newer_records = np.asarray(
            [self.records[0], self.records[2]], dtype=HuffmanCodedPosAndEval)
        newer_records[0]['eval'] = 99
        newer = os.path.join(self.temp_dir.name, 'new.hcpe')
        newer_records.tofile(newer)
        loader = HcpeDataLoader(
            [older, newer], 1, torch.device('cpu'), deduplicate_positions=True)
        self.assertEqual(len(loader), 3)
        self.assertEqual(loader.duplicate_positions_removed, 1)
        start_position = bytes(self.records[0]['hcp'])
        surviving = [row for row in loader.data
                     if bytes(row['hcp']) == start_position]
        self.assertEqual(len(surviving), 1)
        self.assertEqual(int(surviving[0]['eval']), 99)
        loader.executor.shutdown(wait=True)

    def test_recency_decay_weights_newer_files_more_heavily(self):
        older = self.write_file('old.hcpe', [0, 1])
        newer = self.write_file('new.hcpe', [2, 3])
        loader = HcpeDataLoader(
            [older, newer], 1, torch.device('cpu'), recency_decay=0.5)
        source_weights = np.bincount(
            loader.source_indices,
            weights=loader.sampling_probabilities)
        self.assertTrue(np.allclose(source_weights, [1 / 3, 2 / 3]))
        loader.executor.shutdown(wait=True)

    def test_recency_decay_must_be_between_zero_and_one(self):
        path = self.write_file('data.hcpe', [0])
        with self.assertRaisesRegex(ValueError, 'recency_decay'):
            HcpeDataLoader(path, 1, torch.device('cpu'), recency_decay=0)


if __name__ == '__main__':
    unittest.main()
