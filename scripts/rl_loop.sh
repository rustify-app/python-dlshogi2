#!/bin/bash
# Reinforcement-learning self-play loop.
#
# Each iteration:
#   1. generates self-play games with the current model,
#   2. fine-tunes the model on the freshly generated data (resuming from the
#      current checkpoint),
#   3. promotes the candidate only when a paired SPRT accepts it over current.
#
# Start from a supervised-pretrained checkpoint for a warm start. Usage:
#
#   ./scripts/rl_loop.sh checkpoints/checkpoint.pth
#
# Tune the loop with the environment variables below.
set -e

INIT_MODEL="${1:?usage: rl_loop.sh <initial_checkpoint.pth>}"

ITERATIONS="${ITERATIONS:-20}"     # number of self-play/train cycles
GAMES="${GAMES:-1000}"             # self-play games per iteration (across all workers)
PLAYOUTS="${PLAYOUTS:-400}"        # MCTS playouts per move
WORKERS="${WORKERS:-8}"            # parallel self-play workers (saturate the GPU)
SELFPLAY_BATCHSIZE="${SELFPLAY_BATCHSIZE:-32}"  # inference batch size per worker
EPOCHS="${EPOCHS:-1}"              # training epochs per iteration
BATCHSIZE="${BATCHSIZE:-1024}"     # training batch size
LR="${LR:-0.002}"                  # lower LR than SL: fine-tuning
VAL_LAMBDA="${VAL_LAMBDA:-0.5}"    # blend game result with bootstrapped value
GPU="${GPU:-0}"
WORKDIR="${WORKDIR:-rl}"           # where iteration artifacts are written
PYTHON="${PYTHON:-python}"         # python interpreter (set to .venv/bin/python on Vast.ai)
METRICS_DIR="${METRICS_DIR:-$WORKDIR/metrics}"  # structured run metrics
PROMOTION_ENABLED="${PROMOTION_ENABLED:-1}"
PROMOTION_GAMES="${PROMOTION_GAMES:-100}"
PROMOTION_PLAYOUTS="${PROMOTION_PLAYOUTS:-$PLAYOUTS}"
PROMOTION_ELO0="${PROMOTION_ELO0:-0}"
PROMOTION_ELO1="${PROMOTION_ELO1:-5}"
PROMOTION_ALPHA="${PROMOTION_ALPHA:-0.05}"
PROMOTION_BETA="${PROMOTION_BETA:-0.05}"
PROMOTION_PLIES="${PROMOTION_PLIES:-12}"
PROMOTION_TOP_K="${PROMOTION_TOP_K:-8}"

if [ "$PROMOTION_ENABLED" != 0 ] && [ "$PROMOTION_ENABLED" != 1 ]; then
    echo "PROMOTION_ENABLED must be 0 or 1" >&2
    exit 2
fi
if [ "$PROMOTION_ENABLED" = 1 ]; then
    if ! [[ "$PROMOTION_GAMES" =~ ^[0-9]+$ ]] || [ "$PROMOTION_GAMES" -lt 2 ] || [ $((PROMOTION_GAMES % 2)) -ne 0 ]; then
        echo "PROMOTION_GAMES must be a positive even number" >&2
        exit 2
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$WORKDIR" "$METRICS_DIR"
CURRENT="$INIT_MODEL"

# ループ全体を1つのrunとして記録する (各イテレーションはこのrun配下のイベント)
RL_LOG="$METRICS_DIR/rl.jsonl"
RL_RUN_ID="${RL_RUN_ID:-rl-$(date +%Y%m%d-%H%M%S)}"
"$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type run --kind rl --run-id "$RL_RUN_ID" \
    --set init_model="$INIT_MODEL" --set iterations="$ITERATIONS" \
    --set games="$GAMES" --set playouts="$PLAYOUTS" --set workers="$WORKERS" \
    --set epochs="$EPOCHS" --set batchsize="$BATCHSIZE" --set lr="$LR" \
    --set val_lambda="$VAL_LAMBDA" --set workdir="$WORKDIR" \
    --set promotion_enabled="$PROMOTION_ENABLED" --set promotion_games="$PROMOTION_GAMES" \
    --set promotion_playouts="$PROMOTION_PLAYOUTS" --set promotion_elo0="$PROMOTION_ELO0" \
    --set promotion_elo1="$PROMOTION_ELO1" --set promotion_alpha="$PROMOTION_ALPHA" \
    --set promotion_beta="$PROMOTION_BETA" > /dev/null

for i in $(seq 1 "$ITERATIONS"); do
    echo "=== RL iteration $i / $ITERATIONS (model: $CURRENT) ==="
    DATA="$WORKDIR/selfplay-$(printf '%03d' "$i").hcpe"
    NEXT="$WORKDIR/checkpoint-$(printf '%03d' "$i").pth"
    ITER_STARTED=$(date +%s)
    PARENT_MODEL="$CURRENT"

    if [ -s "$NEXT" ]; then
        echo "candidate checkpoint already exists; skipping self-play and training"
    else
        "$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type event --event iteration_start \
            --run-id "$RL_RUN_ID" --set iteration="$i" --set model="$CURRENT" \
            --set data="$DATA" > /dev/null

        if [ -s "$DATA" ]; then
            echo "[1/2] reusing existing self-play data -> $DATA"
        else
            echo "[1/2] parallel self-play ($WORKERS workers) -> $DATA"
            WORKERS="$WORKERS" GAMES="$GAMES" PLAYOUTS="$PLAYOUTS" \
                BATCHSIZE="$SELFPLAY_BATCHSIZE" GPU="$GPU" PYTHON="$PYTHON" \
                METRICS_PREFIX="$METRICS_DIR/selfplay-$(printf '%03d' "$i")" ITERATION="$i" \
                "$SCRIPT_DIR/selfplay_parallel.sh" "$CURRENT" "$DATA"
        fi

        echo "[2/2] train -> $NEXT"
        "$PYTHON" -m pydlshogi2.train "$WORKDIR"/selfplay-*.hcpe "$DATA" \
            --resume "$CURRENT" \
            --epoch "$EPOCHS" \
            --batchsize "$BATCHSIZE" \
            --lr "$LR" \
            --val_lambda "$VAL_LAMBDA" \
            --gpu "$GPU" \
            --checkpoint "$NEXT" \
            --metrics "$METRICS_DIR/train-$(printf '%03d' "$i").jsonl" \
            --run_id "$RL_RUN_ID-train-$(printf '%03d' "$i")"
    fi

    PROMOTION_DECISION=disabled
    PROMOTED=true
    MATCH_METRICS=
    if [ "$PROMOTION_ENABLED" != 0 ]; then
        PROMOTED=false
        OPENING="$WORKDIR/promotion-openings-$(printf '%03d' "$i").txt"
        if [ ! -s "$OPENING" ]; then
            echo "sampling promotion openings -> $OPENING"
            BOOK_LINES=$(( (PROMOTION_GAMES + 1) / 2 + 8 ))
            "$PYTHON" utils/make_opening_book.py "$CURRENT" "$OPENING" \
                --lines "$BOOK_LINES" --plies "$PROMOTION_PLIES" \
                --top-k "$PROMOTION_TOP_K" --gpu "$GPU" --seed "$i"
        fi

        MATCH_METRICS="$METRICS_DIR/promotion-$(printf '%03d' "$i")-$(date +%Y%m%d-%H%M%S)-$$.jsonl"
        echo "[3/3] comparing candidate with current using paired SPRT"
        if "$PYTHON" -m pydlshogi2.match \
            --engine1 "$SCRIPT_DIR/usi_engine.sh" \
            --engine2 "$SCRIPT_DIR/usi_engine.sh" \
            --name1 "candidate-$i" --name2 "current-$i" \
            --options1 "modelfile=$NEXT" --options2 "modelfile=$CURRENT" \
            --games "$PROMOTION_GAMES" --playouts "$PROMOTION_PLAYOUTS" \
            --opening "$OPENING" --sprt \
            --elo0 "$PROMOTION_ELO0" --elo1 "$PROMOTION_ELO1" \
            --alpha "$PROMOTION_ALPHA" --beta "$PROMOTION_BETA" \
            --experiment "rl-promotion-$i" --metrics "$MATCH_METRICS"; then
            MATCH_STATUS=0
        else
            MATCH_STATUS=$?
        fi
        PROMOTION_DECISION=$("$PYTHON" -m pydlshogi2.promotion "$MATCH_METRICS")
        if [ "$PROMOTION_DECISION" = accept ]; then
            PROMOTED=true
            CURRENT="$NEXT"
            echo "candidate accepted; promoted $CURRENT"
        else
            echo "candidate not promoted (SPRT: $PROMOTION_DECISION; match exit: $MATCH_STATUS)"
        fi
    else
        CURRENT="$NEXT"
        echo "promotion gate disabled; promoted $CURRENT"
    fi

    DATA_BYTES=0
    if [ -s "$DATA" ]; then
        DATA_BYTES=$(stat -c%s "$DATA")
    fi
    "$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type event --event promotion_decision \
        --run-id "$RL_RUN_ID" --set iteration="$i" --set parent_model="$PARENT_MODEL" \
        --set candidate="$NEXT" --set decision="$PROMOTION_DECISION" \
        --set promoted="$PROMOTED" --set match_metrics="$MATCH_METRICS" > /dev/null
    "$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type metric \
        --run-id "$RL_RUN_ID" --set scope=iteration --set iteration="$i" \
        --set model="$CURRENT" --set candidate="$NEXT" --set data="$DATA" \
        --set data_bytes="$DATA_BYTES" --set promotion_decision="$PROMOTION_DECISION" \
        --set promoted="$PROMOTED" \
        --set seconds="$(( $(date +%s) - ITER_STARTED ))" > /dev/null
done

"$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type event --event run_end \
    --run-id "$RL_RUN_ID" --set status=completed --set final_model="$CURRENT" > /dev/null

echo "RL loop finished. Final model: $CURRENT"
