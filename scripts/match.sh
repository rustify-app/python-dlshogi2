#!/bin/bash
# Run an engine-vs-engine match in the background, the way a long experiment
# actually gets run: it survives an SSH disconnect and writes everything it
# writes metrics and logs for later analysis.
#
#   ./scripts/match.sh EXP-001 --engine1 ./scripts/usi_engine.sh --engine2 ../wt-main/scripts/usi_engine.sh \
#              --games 100 --byoyomi 1000 --sprt
#
# The first argument is an experiment identifier used in the log and metrics
# filenames.
#
# Tunables (environment variables):
#   PYTHON       interpreter                          (default: .venv/bin/python if present)
#   METRICS_DIR  where the JSONL goes                 (default: metrics)
#   OPENING      opening book                         (default: openings.txt if present)
#   ISSUE        GitHub issue number to record
#   FOREGROUND   set to 1 to run in the foreground
set -e

EXPERIMENT="${1:?usage: match.sh <EXPERIMENT-ID> [match args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "$PYTHON" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON=python3
    fi
fi

METRICS_DIR="${METRICS_DIR:-metrics}"
OPENING="${OPENING:-openings.txt}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$METRICS_DIR" logs

args=(--experiment "$EXPERIMENT"
      --metrics "$METRICS_DIR/match-${EXPERIMENT}-${STAMP}.jsonl")
[ -n "$ISSUE" ] && args+=(--issue "$ISSUE")
# 定跡がなければ全局が同じ将棋になる。あるものは黙って使う
if [ -f "$OPENING" ]; then
    args+=(--opening "$OPENING")
fi

LOG="logs/match-${EXPERIMENT}-${STAMP}.log"
echo "experiment : $EXPERIMENT"
echo "metrics    : $METRICS_DIR/match-${EXPERIMENT}-${STAMP}.jsonl"
echo "log        : $LOG"

if [ "$FOREGROUND" = 1 ]; then
    # exec only replaces the pipeline subshell, so exit here to avoid starting
    # the same match again in the background below.
    "$PYTHON" -m pydlshogi2.match "${args[@]}" "$@" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}"
fi

nohup "$PYTHON" -m pydlshogi2.match "${args[@]}" "$@" > "$LOG" 2>&1 &
echo $! > "logs/match-${EXPERIMENT}.pid"
echo "pid        : $(cat "logs/match-${EXPERIMENT}.pid")"
echo
echo "follow with : tail -f $LOG"
echo "stop with   : kill \$(cat logs/match-${EXPERIMENT}.pid)"
