#!/bin/sh
# USI engine launcher that runs the MCTS player *from this checkout*.
#
# `scripts/mcts_player.sh` starts the PyInstaller build in `dist/`; this one starts the
# Python module in place, which is what the match arena needs: pointing it at
# two different git worktrees is how a branch gets played against `main`.
#
#   python -m pydlshogi2.match --engine1 /path/to/branch/scripts/usi_engine.sh \
#                              --engine2 /path/to/main/scripts/usi_engine.sh ...
#
# Running with the working directory set to the checkout is what makes the
# local `pydlshogi2/` win over any installed copy of the package, so an
# editable install shared between worktrees does not silently make both sides
# play the same code.  `PYTHON` selects the interpreter; a `.venv` next to this
# script is used when it exists.
set -e
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

exec "$PYTHON" -m pydlshogi2.player.mcts_player "$@"
