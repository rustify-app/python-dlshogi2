"""Sample start positions from CSA game records for teacher data generation.

Each kept game contributes up to --per-game positions taken at random plies.
The output has one SFEN per line (without the leading "sfen" keyword).
"""
import argparse
import glob
import os
import random

from cshogi import CSA, Board


def iter_csa_files(csa_dir):
    pattern = os.path.join(csa_dir, '**', '*.csa')
    return glob.iglob(pattern, recursive=True)


def sample_game(kif, rng, min_ply, max_ply, per_game):
    """Return SFENs reached in one game at up to per_game random plies."""
    last = min(max_ply, len(kif.moves))
    if last < min_ply:
        return []
    plies = set(rng.sample(range(min_ply, last + 1),
                           min(per_game, last - min_ply + 1)))
    board = Board(kif.sfen)
    sfens = []
    for ply in range(last + 1):
        if ply in plies:
            sfens.append(board.sfen())
        if ply == last:
            break
        move = kif.moves[ply]
        if not board.is_legal(move):
            break
        board.push(move)
    return sfens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csa_dir')
    parser.add_argument('output')
    parser.add_argument('--min-rating', type=int, default=3000,
                        help='skip games where either player is rated lower')
    parser.add_argument('--min-moves', type=int, default=40)
    parser.add_argument('--min-ply', type=int, default=0)
    parser.add_argument('--max-ply', type=int, default=60)
    parser.add_argument('--per-game', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    games = 0
    with open(args.output, 'w') as out:
        for path in iter_csa_files(args.csa_dir):
            try:
                kifs = CSA.Parser.parse_file(path)
            except Exception:
                continue
            for kif in kifs:
                if len(kif.moves) < args.min_moves:
                    continue
                if args.min_rating > 0 and (
                        not kif.ratings or min(kif.ratings) < args.min_rating):
                    continue
                games += 1
                for sfen in sample_game(kif, rng, args.min_ply, args.max_ply,
                                        args.per_game):
                    # Ignore the move counter when checking for duplicates.
                    key = sfen.rsplit(' ', 1)[0]
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(sfen + '\n')
    print('games', games, 'positions', len(seen))


if __name__ == '__main__':
    main()
