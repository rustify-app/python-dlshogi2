"""Fit the HCPE evaluation-to-win-probability temperature on held-out games."""
import argparse
import json
import math

import numpy as np
from cshogi import Board, HuffmanCodedPosAndEval

from pydlshogi2.features import BLACK, make_result


def log_loss(eval_stm, outcomes, coefficient):
    """Return mean binary cross-entropy for side-to-move outcomes."""
    logits = np.asarray(eval_stm, dtype=np.float64) / coefficient
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    return float(-np.mean(outcomes * np.log(probabilities)
                           + (1.0 - outcomes) * np.log1p(-probabilities)))


def fit_eval_coefficient(eval_stm, outcomes, lower=50.0, upper=5000.0):
    """Find the coefficient minimizing log loss using golden-section search."""
    eval_stm = np.asarray(eval_stm, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if eval_stm.ndim != 1 or outcomes.shape != eval_stm.shape or not len(eval_stm):
        raise ValueError('evaluation and outcome arrays must be non-empty 1-D arrays of equal length')
    if not np.all(np.isfinite(eval_stm)) or not np.all(np.isfinite(outcomes)):
        raise ValueError('evaluation and outcome arrays must be finite')
    if np.any((outcomes < 0) | (outcomes > 1)):
        raise ValueError('outcomes must be in [0, 1]')

    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1 = log_loss(eval_stm, outcomes, math.exp(x1))
    f2 = log_loss(eval_stm, outcomes, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = log_loss(eval_stm, outcomes, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = log_loss(eval_stm, outcomes, math.exp(x2))
    return math.exp((left + right) / 2.0)


def calibrate(path, max_positions=None, seed=0, exclude_paths=()):
    data = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
    excluded = set()
    for exclude_path in exclude_paths:
        excluded_data = np.fromfile(exclude_path, dtype=HuffmanCodedPosAndEval)
        excluded.update(bytes(hcp) for hcp in excluded_data['hcp'])
    if excluded:
        keep = np.fromiter((bytes(hcp) not in excluded for hcp in data['hcp']),
                           dtype=np.bool_, count=len(data))
        data = data[keep]
    if max_positions is not None and max_positions < len(data):
        indices = np.random.default_rng(seed).choice(len(data), max_positions,
                                                     replace=False)
        data = data[indices]
    board = Board()
    eval_stm = np.empty(len(data), dtype=np.float64)
    outcomes = np.empty(len(data), dtype=np.float64)
    for index, record in enumerate(data):
        board.set_hcp(record['hcp'])
        raw_eval = float(record['eval'])
        eval_stm[index] = raw_eval if board.turn == BLACK else -raw_eval
        outcomes[index] = make_result(record['gameResult'], board.turn)
    coefficient = fit_eval_coefficient(eval_stm, outcomes)
    return {
        'positions': len(data),
        'excluded_positions': len(excluded),
        'eval_coef_default': 600.0,
        'eval_coef_calibrated': coefficient,
        'log_loss_default': log_loss(eval_stm, outcomes, 600.0),
        'log_loss_calibrated': log_loss(eval_stm, outcomes, coefficient),
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='held-out HCPE data; keep it out of training')
    parser.add_argument('--max-positions', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exclude-data', action='append', default=[],
                        help='HCPE file whose positions must be excluded; repeat as needed')
    parser.add_argument('--report', default=None, help='optional JSON report path')
    args = parser.parse_args()
    if args.max_positions is not None and args.max_positions < 1:
        parser.error('--max-positions must be positive')
    result = calibrate(args.data, args.max_positions, args.seed, args.exclude_data)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as output:
            output.write(encoded + '\n')


if __name__ == '__main__':
    main()
