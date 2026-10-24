import json
import os
import tempfile
import unittest

from pydlshogi2.promotion import read_sprt_decision


class ReadSprtDecisionTest(unittest.TestCase):
    def write_metrics(self, records):
        fd, path = tempfile.mkstemp(text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_reads_accept_from_summary(self):
        path = self.write_metrics([
            {'scope': 'game', 'sprt_decision': 'continue'},
            {'scope': 'summary', 'sprt_decision': 'accept'},
        ])
        self.assertEqual(read_sprt_decision(path), 'accept')

    def test_reads_latest_summary(self):
        path = self.write_metrics([
            {'scope': 'summary', 'sprt_decision': 'accept'},
            {'scope': 'summary', 'sprt_decision': 'continue'},
        ])
        self.assertEqual(read_sprt_decision(path), 'continue')

    def test_rejects_missing_summary(self):
        path = self.write_metrics([{'scope': 'game'}])
        with self.assertRaisesRegex(ValueError, 'no summary'):
            read_sprt_decision(path)

    def test_rejects_invalid_decision(self):
        path = self.write_metrics([{'scope': 'summary'}])
        with self.assertRaisesRegex(ValueError, 'valid SPRT decision'):
            read_sprt_decision(path)


if __name__ == '__main__':
    unittest.main()
