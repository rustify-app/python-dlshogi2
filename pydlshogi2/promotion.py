"""Read a match result and decide whether an RL candidate may be promoted."""
import json
import sys


DECISIONS = frozenset(('accept', 'reject', 'continue'))


def read_sprt_decision(path):
    """Return the last SPRT decision in a match metrics JSONL file.

    An absent or invalid summary is an error so callers fail closed and do not
    promote a candidate when the comparison did not finish correctly.
    """
    summary = None
    with open(path, encoding='utf-8') as metrics_file:
        for line_number, line in enumerate(metrics_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    'invalid JSON on line {}'.format(line_number)) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    'expected a JSON object on line {}'.format(line_number))
            if record.get('scope') == 'summary':
                summary = record

    if summary is None:
        raise ValueError('match metrics have no summary record')
    decision = summary.get('sprt_decision')
    if decision not in DECISIONS:
        raise ValueError('match summary has no valid SPRT decision')
    return decision


def main():
    if len(sys.argv) != 2:
        print('usage: python -m pydlshogi2.promotion <match-metrics.jsonl>',
              file=sys.stderr)
        return 2
    try:
        print(read_sprt_decision(sys.argv[1]))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
