"""Generate HCPE teacher data by letting a USI engine play itself.

Every game starts from a random position of the start-position file, optionally
followed by a few random legal moves for diversity. Each position is searched
with a fixed node budget; its best move and score (side to move) are recorded
and the game result is filled in when the game ends.

Several workers run in parallel, each with its own single-threaded engine,
and append finished games to their own output file so that an interrupted run
keeps everything written so far.

Example:
    python utils/gen_teacher.py starts.sfen out_dir --engine ~/yo/YaneuraOu-by-gcc \
        --workers 4 --nodes 10000 --positions 1000000
"""
import argparse
import multiprocessing as mp
import os
import random
import socket
import sys
import time

import numpy as np
from cshogi import (BLACK, BLACK_WIN, DRAW, WHITE_WIN, Board,
                    HuffmanCodedPosAndEval, REPETITION_DRAW, REPETITION_LOSE,
                    REPETITION_WIN, REPETITION_SUPERIOR, REPETITION_INFERIOR,
                    move16)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_multipv_targets import MATE_SCORE, UsiEngine  # noqa: E402


class SelfPlayEngine(UsiEngine):
    """USI client returning (score, bestmove) for a single PV search."""

    def new_game(self):
        self._send('usinewgame')

    def best(self, board):
        self._send('position sfen {}'.format(board.sfen()))
        self._send('go nodes {}'.format(self.nodes))
        score = None
        deadline = time.monotonic() + self.timeout
        while True:
            line = self._next_line(deadline)
            if line.startswith('info ') and ' score ' in line:
                parsed = self._parse_score(line)
                if parsed is not None:
                    score = parsed
            elif line.startswith('bestmove '):
                return score, line.split()[1]

    @staticmethod
    def _parse_score(line):
        tokens = line.split()
        try:
            index = tokens.index('score')
            kind, value = tokens[index + 1], int(tokens[index + 2])
        except (ValueError, IndexError):
            return None
        if kind == 'cp':
            return value
        if kind == 'mate':
            return MATE_SCORE if value > 0 else -MATE_SCORE
        return None


def result_for(turn, side_to_move_wins):
    """Game result constant when the side to move wins (True) or loses."""
    black_wins = (turn == BLACK) == side_to_move_wins
    return BLACK_WIN if black_wins else WHITE_WIN


def play_game(engine, board, rng, args):
    """Play one game; return (records, result) with records ready for HCPE."""
    records = []
    for _ in range(rng.randint(0, args.random_moves)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    if board.is_game_over():
        return [], None

    engine.new_game()
    result = DRAW
    for _ in range(args.max_plies):
        if board.is_game_over():
            result = result_for(board.turn, False)
            break
        draw = board.is_draw(16)
        if draw == REPETITION_DRAW:
            result = DRAW
            break
        if draw in (REPETITION_WIN, REPETITION_SUPERIOR):
            result = result_for(board.turn, True)
            break
        if draw in (REPETITION_LOSE, REPETITION_INFERIOR):
            result = result_for(board.turn, False)
            break

        score, usi = engine.best(board)
        if usi == 'resign':
            result = result_for(board.turn, False)
            break
        if usi == 'win':
            result = result_for(board.turn, True)
            break
        if score is None:
            score = 0
        move = board.move_from_usi(usi)
        if not board.is_legal(move):
            return [], None

        record = np.zeros(1, HuffmanCodedPosAndEval)[0]
        board.to_hcp(record['hcp'])
        record['eval'] = max(-32767, min(32767, score))
        record['bestMove16'] = move16(move)
        records.append(record)

        if abs(score) >= args.resign_score:
            result = result_for(board.turn, score > 0)
            break
        board.push(move)
    return records, result


def worker(index, args, counter, lock):
    rng = random.Random((args.seed << 16) + index)
    starts = load_starts(args.start_positions)
    name = '{}-{}-{}.hcpe'.format(args.prefix, os.getpid(), index)
    path = os.path.join(args.output_dir, name)
    engine = SelfPlayEngine(args.engine, args.engine_dir, 1, args.nodes,
                            1, args.hash_mb, args.timeout)
    board = Board()
    try:
        with open(path, 'ab') as out:
            while True:
                with lock:
                    if args.positions and counter.value >= args.positions:
                        return
                board.set_sfen(rng.choice(starts))
                try:
                    records, result = play_game(engine, board, rng, args)
                except (TimeoutError, RuntimeError) as exc:
                    print('worker', index, 'restarting engine:', exc, flush=True)
                    engine.close()
                    engine = SelfPlayEngine(args.engine, args.engine_dir, 1,
                                            args.nodes, 1, args.hash_mb,
                                            args.timeout)
                    continue
                if not records or (result == DRAW and args.skip_draws):
                    continue
                data = np.array(records, dtype=HuffmanCodedPosAndEval)
                data['gameResult'] = result
                data.tofile(out)
                out.flush()
                with lock:
                    counter.value += len(data)
    finally:
        engine.close()


def load_starts(path):
    with open(path) as f:
        starts = [line.strip() for line in f if line.strip()]
    starts = [s[5:] if s.startswith('sfen ') else s for s in starts]
    if not starts:
        raise SystemExit('no start positions in ' + path)
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('start_positions', help='file with one SFEN per line')
    parser.add_argument('output_dir')
    parser.add_argument('--engine', required=True)
    parser.add_argument('--engine-dir', default=None)
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument('--nodes', type=int, default=10000)
    parser.add_argument('--positions', type=int, default=0,
                        help='stop after this many positions; 0 runs forever')
    parser.add_argument('--random-moves', type=int, default=4,
                        help='maximum random legal moves played after the start position')
    parser.add_argument('--max-plies', type=int, default=320)
    parser.add_argument('--resign-score', type=int, default=3000,
                        help='adjudicate the game when |score| reaches this value')
    parser.add_argument('--skip-draws', action='store_true')
    parser.add_argument('--hash-mb', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--prefix', default=socket.gethostname())
    args = parser.parse_args()

    args.engine = os.path.abspath(os.path.expanduser(args.engine))
    args.engine_dir = os.path.abspath(os.path.expanduser(
        args.engine_dir or os.path.dirname(args.engine)))
    os.makedirs(args.output_dir, exist_ok=True)

    counter = mp.Value('q', 0)
    lock = mp.Lock()
    processes = [mp.Process(target=worker, args=(i, args, counter, lock), daemon=True)
                 for i in range(args.workers)]
    for p in processes:
        p.start()
    started = time.time()
    last_report = started

    def report():
        elapsed = time.time() - started
        print('positions {} elapsed {:.0f}s rate {:.1f}/s'.format(
            counter.value, elapsed, counter.value / max(elapsed, 1e-9)), flush=True)

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(1)
            if time.time() - last_report >= 60:
                report()
                last_report = time.time()
    except KeyboardInterrupt:
        pass
    report()
    for p in processes:
        p.join(timeout=5)


if __name__ == '__main__':
    main()
