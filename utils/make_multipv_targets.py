"""Create sparse policy targets from a USI engine's MultiPV evaluations.

Each target row is keyed by the HCPE position bytes, so it can be used with
one or more HCPE training files without changing their on-disk format.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
from cshogi import Board, HuffmanCodedPosAndEval, move_to_usi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2.features import make_move_label  # noqa: E402


MATE_SCORE = 30000


class UsiEngine:
    """Small synchronous client for a line-oriented USI engine."""

    def __init__(self, executable, cwd, multipv, nodes, threads, hash_mb,
                 timeout):
        self.timeout = timeout
        command = [executable]
        if os.name == 'posix':
            command = ['stdbuf', '-oL', '-eL'] + command
        self.process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

        self._send('usi')
        self._wait_for('usiok')
        self._send('setoption name Threads value {}'.format(threads))
        self._send('setoption name MultiPV value {}'.format(multipv))
        self._send('setoption name USI_Hash value {}'.format(hash_mb))
        self._send('setoption name USI_OwnBook value false')
        self._send('setoption name BookFile value no_book')
        self._send('isready')
        self._wait_for('readyok')
        self.nodes = nodes

    def _read_output(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip())
        self.lines.put(None)

    def _send(self, command):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def _next_line(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('USI engine response timed out')
        try:
            line = self.lines.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError('USI engine response timed out') from exc
        if line is None:
            raise RuntimeError('USI engine exited unexpectedly')
        return line

    def _wait_for(self, prefix):
        deadline = time.monotonic() + self.timeout
        while True:
            line = self._next_line(deadline)
            if line.startswith(prefix):
                return

    @staticmethod
    def _parse_info(line):
        tokens = line.split()
        try:
            rank = int(tokens[tokens.index('multipv') + 1])
            score_index = tokens.index('score')
            score_kind = tokens[score_index + 1]
            score_value = int(tokens[score_index + 2])
            pv = tokens[tokens.index('pv') + 1:]
        except (ValueError, IndexError):
            return None
        if score_kind == 'cp':
            score = score_value
        elif score_kind == 'mate':
            score = (MATE_SCORE - min(abs(score_value), 1000)
                     if score_value > 0 else
                     -MATE_SCORE + min(abs(score_value), 1000))
        else:
            return None
        if not pv or pv[0] == '(none)':
            return None
        return rank, score, pv[0]

    def search(self, board):
        self._send('position sfen {}'.format(board.sfen()))
        self._send('go nodes {}'.format(self.nodes))
        candidates = {}
        deadline = time.monotonic() + self.timeout
        while True:
            line = self._next_line(deadline)
            if line.startswith('info '):
                parsed = self._parse_info(line)
                if parsed is not None:
                    rank, score, move = parsed
                    candidates[rank] = (score, move)
            elif line.startswith('bestmove '):
                if not candidates:
                    raise RuntimeError('USI engine returned no MultiPV candidates')
                return candidates

    def close(self):
        if self.process.poll() is None:
            try:
                self._send('quit')
                self.process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)


def softmax(scores, temperature_cp):
    scaled = (np.asarray(scores, dtype=np.float64) - max(scores)) / temperature_cp
    probabilities = np.exp(np.clip(scaled, -80.0, 0.0))
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_hcpe', help='source positions in HCPE format')
    parser.add_argument('output_npz', help='sparse MultiPV targets')
    parser.add_argument('--engine', required=True, help='USI engine executable')
    parser.add_argument('--engine-dir', default=None,
                        help='working directory containing engine evaluation files')
    parser.add_argument('--multipv', type=int, default=5,
                        help='number of principal variations to collect')
    parser.add_argument('--nodes', type=int, default=2048,
                        help='search nodes for each position')
    parser.add_argument('--temperature-cp', type=float, default=100.0,
                        help='softmax temperature for engine scores')
    parser.add_argument('--max-positions', type=int, default=1000,
                        help='maximum positions to annotate; 0 uses every position')
    parser.add_argument('--positions-output', default=None,
                        help='optional HCPE file containing the sampled input positions')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--hash-mb', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='engine response timeout in seconds')
    args = parser.parse_args()

    if args.multipv < 1 or args.nodes < 1 or args.temperature_cp <= 0:
        parser.error('multipv, nodes, and temperature-cp must be positive')
    if args.max_positions < 0 or args.threads < 1 or args.hash_mb < 1:
        parser.error('max-positions must be non-negative; threads and hash-mb must be positive')

    executable = os.path.abspath(args.engine)
    if not os.path.isfile(executable):
        parser.error('engine executable does not exist: {}'.format(executable))
    engine_dir = os.path.abspath(args.engine_dir or os.path.dirname(executable))
    records = np.fromfile(args.input_hcpe, dtype=HuffmanCodedPosAndEval)
    if len(records) == 0:
        parser.error('input HCPE file contains no positions')

    rng = np.random.default_rng(args.seed)
    if args.max_positions and len(records) > args.max_positions:
        indices = np.sort(rng.choice(len(records), args.max_positions, replace=False))
        records = records[indices]
    else:
        indices = np.arange(len(records))
    if args.positions_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.positions_output)),
                    exist_ok=True)
        records.tofile(args.positions_output)

    board = Board()
    engine = UsiEngine(executable, engine_dir, args.multipv, args.nodes,
                       args.threads, args.hash_mb, args.timeout)
    seen = set()
    hcp_rows = []
    label_rows = []
    probability_rows = []
    score_rows = []
    source_indices = []
    try:
        for source_index, record in zip(indices, records):
            hcp = bytes(record['hcp'])
            if hcp in seen:
                continue
            seen.add(hcp)
            board.set_hcp(record['hcp'])
            legal_moves = {move_to_usi(move): move for move in board.legal_moves}
            candidates = engine.search(board)
            ranked = []
            used_labels = set()
            for rank in sorted(candidates):
                score, usi_move = candidates[rank]
                move = legal_moves.get(usi_move)
                if move is None:
                    continue
                label = make_move_label(move, board.turn)
                if label in used_labels:
                    continue
                used_labels.add(label)
                ranked.append((label, score))
            if not ranked:
                continue

            labels = np.full(args.multipv, -1, dtype=np.int64)
            scores = np.zeros(args.multipv, dtype=np.int32)
            probabilities = np.zeros(args.multipv, dtype=np.float32)
            count = min(len(ranked), args.multipv)
            labels[:count] = [item[0] for item in ranked[:count]]
            scores[:count] = [item[1] for item in ranked[:count]]
            probabilities[:count] = softmax(scores[:count], args.temperature_cp)
            hcp_rows.append(np.frombuffer(hcp, dtype=np.uint8).copy())
            label_rows.append(labels)
            probability_rows.append(probabilities)
            score_rows.append(scores)
            source_indices.append(source_index)
    finally:
        engine.close()

    if not hcp_rows:
        raise RuntimeError('no legal MultiPV targets were collected')
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        hcp=np.stack(hcp_rows),
        labels=np.stack(label_rows),
        probabilities=np.stack(probability_rows),
        scores_cp=np.stack(score_rows),
        source_indices=np.asarray(source_indices, dtype=np.int64))
    print('wrote {} MultiPV targets from {} sampled positions to {}'.format(
        len(hcp_rows), len(indices), args.output_npz))


if __name__ == '__main__':
    main()
