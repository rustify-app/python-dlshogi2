# python-dlshogi2

Deep learning shogi AI engine using a policy-value network and Monte Carlo Tree Search (MCTS), inspired by the AlphaGo Zero approach.


---

## Overview

python-dlshogi2 trains a ResNet-based neural network on shogi game records, then uses the trained model to play shogi via MCTS. It communicates with shogi GUIs using the **USI (Universal Shogi Interface)** protocol.

**Key features:**
- Configurable policy-value ResNet with optional Squeeze-and-Excitation (default: 20 blocks, 256 channels, SE)
- Supervised training with an evaluation-blended value target
- Self-play reinforcement learning loop (AlphaZero / expert-iteration style)
- MCTS with virtual loss, FPU and an AlphaZero-style PUCT term
- Mate detection (configurable root mate search) and draw recognition
- Ponder support (thinking during opponent's turn)
- PyTorch and ONNX (CUDA/TensorRT) inference backends
- Resignation based on configurable win-rate threshold
- Engine-vs-engine match arena with Elo, confidence intervals and SPRT early stopping
- Structured run metrics in JSON Lines format


---

## Requirements

- Python 3.x
- PyTorch
- onnxruntime (for ONNX player)
- numpy
- cshogi
- scikit-learn (for data conversion utilities)

---

## Installation

```bash
pip install cshogi
pip install git+https://github.com/LoveKapibarasan/python-dlshogi2.git
```

---

## Data Preparation

Convert CSA game records to HCPE training data:

```bash
python utils/csa_to_hcpe.py <csa_dir> train.hcpe test.hcpe \
    --filter_moves 50 \
    --filter_rating 3500 \
    --test_ratio 0.1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--filter_moves` | 50 | Minimum number of moves per game |
| `--filter_rating` | 3500 | Minimum player rating |
| `--test_ratio` | 0.1 | Fraction of data used for testing |

---

## Training

```bash
python -m pydlshogi2.train train.hcpe test.hcpe \
    --gpu 0 \
    --epoch 10 \
    --batchsize 1024 \
    --lr 0.01 \
    --checkpoint checkpoints/checkpoint-{epoch:03}.pth
```

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu` | 0 | GPU ID (-1 for CPU) |
| `--epoch` | 10 | Number of training epochs |
| `--batchsize` | 1024 | Mini-batch size |
| `--lr` | 0.01 | Learning rate (SGD with momentum) |
| `--checkpoint` | — | Checkpoint path template |
| `--blocks` / `--channels` / `--fcl` | 20 / 256 / 256 | Network size (ignored when `--resume`) |
| `--no_se` | off | Disable Squeeze-and-Excitation blocks |
| `--val_lambda` | 0.333 | Weight on game outcome vs. search eval in the value target |
| `--eval_coef` | 600 | Sigmoid temperature for eval→win-rate |
| `--amp` | off | bfloat16 autocast (mixed precision) |
| `--compile` | off | Wrap model with `torch.compile` |
| `--save_interval` | 0 | Save a checkpoint every N steps (0 = epoch end only) |
| `--resume` | — | Resume from a checkpoint (model + optimizer + step + architecture) |
| `--metrics` | — | Append structured metrics in JSON Lines format |
| `--run_id` | — | Reuse a run id so a preempted run and its resumes stay one run |

### Interrupting & resuming (preemptible instances)

Training is preemption-safe: on `SIGTERM`/`SIGINT` it checkpoints after the
current step and exits cleanly, and `--save_interval` writes periodic
checkpoints mid-epoch. Resume with the **same checkpoint path**:

```bash
# initial run (writes a single rolling checkpoint)
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --save_interval 1000 --checkpoint checkpoints/latest.pth

# after a preemption, continue from where it stopped
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --resume checkpoints/latest.pth --checkpoint checkpoints/latest.pth
```

`--epoch` is the number of *additional* epochs to run on resume. The optimizer
state, global step and network architecture are all restored from the
checkpoint.

---

## Playing (USI Engine)

### PyTorch player

```bash
python -m pydlshogi2.player.mcts_player
# or
./scripts/mcts_player.sh
```

### ONNX player

The ONNX backend now uses the **same feature representation as the PyTorch
backend**, so it plays models exported from this repository's own checkpoints.
First export a trained checkpoint to ONNX:

```bash
python utils/export_onnx.py checkpoints/checkpoint.pth model/model.onnx
```

Then run the engine:

```bash
python -m pydlshogi2.player.onnx_player
# or
./scripts/onnx_player.sh
```

Export a model from a checkpoint produced by your own training run as shown above.

### USI options

| Option | Default | Description |
|--------|---------|-------------|
| `modelfile` | — | Path to checkpoint or ONNX model |
| `gpu_id` | 0 | GPU ID (-1 for CPU) |
| `batchsize` | 32 | Neural network batch size |
| `resign_threshold` | 1 | Win rate (%) below which to resign |
| `c_puct` | 100 | MCTS exploration constant, as a percentage (100 = 1.0) |
| `c_base` | 19652 | Base of the PUCT log term |
| `fpu_reduction` | 27 | First Play Urgency reduction, as a percentage (27 = 0.27) |
| `temperature` | 100 | Policy softmax temperature, as a percentage (100 = 1.0) |
| `mate_root_ply` | 7 | Depth of the one-shot root mate search (1-31) |
| `playouts` | 1000 | Playouts for a `go` with no time control (used by fixed-playout matches) |
| `time_margin` | 1000 | Time margin in milliseconds |
| `byoyomi_margin` | 100 | Byoyomi margin in milliseconds |
| `pv_interval` | 500 | PV info output interval (ms) |
| `debug` | false | Enable debug output |

Connect the engine to any USI-compatible shogi GUI (e.g., Shogidokoro, ShogiGUI).

---

## Model Architecture

- **Input:** 104 feature planes on a 9×9 board (piece positions + captured pieces for both sides; `FEATURES_NUM`)
- **Backbone:** Convolutional layer → ResNet blocks with optional Squeeze-and-Excitation (default 20 blocks × 256 channels, batch norm, ReLU)
- **Policy head:** 2,187 move outputs (27 planes × 81 squares = 20 directions + 7 drop pieces; `MOVE_LABELS_NUM`)
- **Value head:** Single sigmoid output (estimated win probability)

The architecture (`--blocks`, `--channels`, `--fcl`, `--no_se`) is configurable
and embedded in each checkpoint, so players and the ONNX exporter reconstruct it
automatically. Checkpoints saved before this feature load as a legacy 10×192
SE-free network.

## Self-play Reinforcement Learning

Generate self-play games and improve the model in a loop:

```bash
# one batch of self-play games (single process)
python -m pydlshogi2.selfplay checkpoints/checkpoint.pth selfplay.hcpe \
    --games 1000 --playouts 800 --gpu 0

# parallel self-play (recommended): a single process is CPU-bound on the MCTS
# tree and leaves the GPU idle; N workers sharing the GPU multiply throughput.
WORKERS=8 GAMES=1000 PLAYOUTS=400 GPU=0 \
    ./scripts/selfplay_parallel.sh checkpoints/checkpoint.pth selfplay.hcpe

# full generate -> train -> promote loop (uses parallel self-play internally)
WORKERS=8 ./scripts/rl_loop.sh checkpoints/checkpoint.pth
```

The RL loop honours `WORKERS`, `GAMES`, `PLAYOUTS`, `SELFPLAY_BATCHSIZE`,
`ITERATIONS`, `EPOCHS`, `LR`, `VAL_LAMBDA`, `GPU`, `WORKDIR` and
`METRICS_DIR`. It writes JSONL records to `$WORKDIR/metrics` by default.

## Evaluation: does the change actually help?

Training curves say a model fits its data. They do not say it plays better, and
a change to the *search* moves no loss at all. The only instrument for that is a
match:

```bash
# 1. build an opening book (once). The search is deterministic, so without one
#    every game of the "match" is a replay of the first.
python utils/make_opening_book.py checkpoints/checkpoint.pth openings.txt \
    --lines 64 --plies 12

# 2. put a branch across the board from main
git worktree add ../wt-main main

python -m pydlshogi2.match \
    --engine1 ./scripts/usi_engine.sh          --name1 branch \
    --engine2 ../wt-main/scripts/usi_engine.sh --name2 main \
    --options1 modelfile=checkpoints/checkpoint.pth \
    --options2 modelfile=checkpoints/checkpoint.pth \
    --games 100 --byoyomi 1000 --opening openings.txt --sprt \
    --metrics metrics/match-EXP-001.jsonl --experiment EXP-001 --issue 7
```

```
=== match result: branch vs main ===
games      : 64 (stopped early by SPRT)
W-L-D      : 24-14-26
pairs      : 32 scored as [0, 1/2, 1, 1 1/2, 2] = [1, 5, 12, 11, 3]
Elo        : +54.7 +/- 41.2   95% CI [+14.1, +96.6]
LOS        : 96.4%
SPRT       : llr=+2.981 bounds=[-2.944, 2.944] H0=0 H1=20 -> accept
verdict    : engine1 is stronger; adopt the change
```

`./scripts/match.sh EXP-001 --engine1 ... --engine2 ...` runs the same thing in the
background with the log and metrics paths filled in for you.

Three things are worth knowing before trusting a number:

- **100 games is less than it looks.** Counting games one by one, the 95 %
  interval over 100 games is about **±70 Elo** — "55-45, so it's better" is not a
  result. `--sprt` stops as soon as the answer is clear in either direction, and
  scoring colour-swapped *pairs* (the default) roughly halves the games needed.
- **`--playouts` and `--byoyomi` measure different things.** Fixed playouts
  compare two models independently of speed; fixed time is the only condition
  under which a faster search can show up as strength. A pure speed-up *must*
  score 50 % at fixed playouts — that is the proof it changed no behaviour.
- **The engines must differ.** Both sides are launched from a checkout via
  `scripts/usi_engine.sh`, so comparing branches means pointing them at two git
  worktrees.

`pydlshogi2/rating.py` holds the arithmetic (Elo, confidence intervals, LOS,
SPRT, and a Bradley-Terry fit that puts every checkpoint on one scale); it is
standard-library only. The full method, including how to read the verdict, is in
the wiki's
[対局評価](https://github.com/LoveKapibarasan/python-dlshogi2/wiki/%E5%AF%BE%E5%B1%80%E8%A9%95%E4%BE%A1).


---

## Run metrics

Training, self-play, reinforcement learning and matches can append run records in JSON Lines format with `--metrics`. The reinforcement learning loop writes records under `$WORKDIR/metrics`.

## Running on a GPU server (Vast.ai)

One-shot bootstrap + background jobs on a fresh CUDA instance:

```bash
# 1) provision: venv + CUDA torch + deps + onnxruntime-gpu + editable install
./scripts/vast_setup.sh

# 2) launch a job in the background (auto-runs setup if needed).
#    Logs to logs/<mode>-<timestamp>.log; survives SSH disconnects.
./scripts/vast_run.sh train    train.hcpe test.hcpe --gpu 0 --amp        # supervised
./scripts/vast_run.sh selfplay checkpoints/checkpoint.pth sp.hcpe --games 1000 --gpu 0
./scripts/vast_run.sh rl       checkpoints/checkpoint.pth                # full self-play loop
```

Follow a job with `tail -f logs/<mode>-*.log` and stop it with
`kill $(cat logs/<mode>.pid)`. The RL loop is tunable via environment variables
(`ITERATIONS`, `GAMES`, `PLAYOUTS`, `EPOCHS`, `LR`, `VAL_LAMBDA`, `GPU`,
`WORKDIR`).

## Project Wiki

The Wiki contains concise Japanese references for architecture, search and operation.
## Training Outputs

Training saves PyTorch checkpoints under `checkpoints/` by default. The directory is created when the first checkpoint is saved; no pre-trained PyTorch checkpoint files are tracked in this repository. Use `--resume` with a checkpoint produced by your own training run.

A checkpoint stores the epoch, step count, network weights, optimizer state, and network configuration. The optimizer state is used to resume training.

## License

See [LICENSE](LICENSE).
