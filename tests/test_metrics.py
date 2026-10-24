"""JSONL 計測記録の書き込みテスト。"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pydlshogi2.metrics import MetricsWriter, new_run_id  # noqa: E402


def read_records(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


class NewRunIdTest(unittest.TestCase):
    def test_ids_are_unique_and_sortable(self):
        ids = [new_run_id() for _ in range(5)]
        self.assertEqual(len(set(ids)), 5)
        self.assertTrue(all(len(run_id) == len(ids[0]) for run_id in ids))


class MetricsWriterTest(unittest.TestCase):
    def test_disabled_writer_is_a_noop(self):
        writer = MetricsWriter(None, kind='train', args={'lr': 0.01})
        writer.metric(step=1, train_loss_total=1.0)
        writer.event('checkpoint', path='x')
        writer.close()
        self.assertTrue(writer.run_id)

    def test_writes_run_metric_and_event_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train', args={'lr': 0.01})
            writer.metric(scope='epoch', epoch=1, step=10, train_loss_total=3.0)
            writer.close(status='completed')
            records = read_records(path)

        self.assertEqual([record['type'] for record in records],
                         ['run', 'metric', 'event'])
        self.assertEqual(records[0]['kind'], 'train')
        self.assertEqual(records[0]['args']['lr'], 0.01)
        self.assertEqual(records[-1]['event'], 'run_end')
        self.assertTrue(all(record['run_id'] == writer.run_id
                            for record in records))

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train')
            writer.close()
            writer.close()
            records = read_records(path)
        self.assertEqual(sum(record.get('event') == 'run_end'
                             for record in records), 1)

    def test_non_serialisable_values_are_stringified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train', args={'device': object()})
            writer.close()
            records = read_records(path)
        self.assertIsInstance(records[0]['args']['device'], str)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'nested', 'deeper', 'train.jsonl')
            MetricsWriter(path, kind='train').close()
            self.assertTrue(os.path.exists(path))

    def test_resumed_runs_can_share_an_id(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, 'first.jsonl')
            second_path = os.path.join(directory, 'resume.jsonl')
            MetricsWriter(first_path, kind='train', run_id='fixed-id').close()
            MetricsWriter(second_path, kind='train', run_id='fixed-id').close()
            records = read_records(first_path) + read_records(second_path)
        self.assertEqual({record['run_id'] for record in records}, {'fixed-id'})


class MetricsCliTest(unittest.TestCase):
    def run_cli(self, path, *args):
        subprocess.check_call(
            [sys.executable, '-m', 'pydlshogi2.metrics', path] + list(args),
            cwd=REPO_ROOT, stdout=subprocess.DEVNULL)

    def test_appends_records_from_the_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'rl.jsonl')
            self.run_cli(path, '--type', 'run', '--kind', 'rl',
                         '--run-id', 'rl-1', '--set', 'iterations=20')
            self.run_cli(path, '--type', 'metric', '--run-id', 'rl-1',
                         '--set', 'scope=iteration', '--set', 'iteration=1',
                         '--set', 'checkpoint=rl/checkpoint-001.pth')
            self.run_cli(path, '--type', 'event', '--event', 'run_end',
                         '--run-id', 'rl-1', '--set', 'status=completed')
            records = read_records(path)

        self.assertEqual([record['type'] for record in records],
                         ['run', 'metric', 'event'])
        self.assertEqual(records[0]['iterations'], 20)
        self.assertEqual(records[1]['iteration'], 1)
        self.assertEqual(records[1]['checkpoint'], 'rl/checkpoint-001.pth')
        self.assertEqual(records[2]['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
