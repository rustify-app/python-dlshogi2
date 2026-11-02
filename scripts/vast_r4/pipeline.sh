#!/bin/bash
# r4 on Vast.ai: wait for prep -> fine-tune from r3 -> export ONNX -> match r4 vs r3 -> results in /workspace/out.
# Ends by stopping this instance itself (no GPU billing). The workstation watcher downloads out/ and destroys it.
set -u
cd /workspace
export PYTHONPATH=/workspace/dl
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
vast_stop() { curl -s -m 60 -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a pipeline.log; }
selfstop() { log "self-stop: $1"; for t in 1 2 3; do vast_stop >> pipeline.log 2>&1; sleep 120; done; }
mkdir -p out ckpt

until grep -q "ALL DONE" prep.log && grep -q "converted 2026: position_num" prep.log; do sleep 60; done
log "prep done"; grep converted prep.log | tee -a pipeline.log

n3800=$(( $(cat hcpe/*-3800-train.hcpe | wc -c) / 38 ))
if [ "$n3800" -ge 30000000 ]; then R=3800; else R=3500; fi
N=$(( $(cat hcpe/*-$R-train.hcpe | wc -c) / 38 ))
cat hcpe/*-$R-test.hcpe | head -c $((38*200000)) > test.hcpe
EPOCHS=1
log "using rating>=$R: $N train positions, $EPOCHS epoch(s)"

python -m pydlshogi2.train hcpe/*-$R-train.hcpe test.hcpe --gpu 0 --epoch $EPOCHS --batchsize 1024 \
  --lr 0.02 --lr_schedule cosine --lr_min 0.0002 --warmup_steps 500 --val_lambda 1.0 --amp \
  --resume r3h.pth --checkpoint ckpt/r4-{epoch:03}.pth --save_interval 20000 --eval_interval 2000 \
  --log out/train.log --metrics out/train.jsonl > out/train.stdout 2>&1 || { log "train failed"; tail -30 out/train.stdout >> pipeline.log; cp pipeline.log out/; touch out/DONE; selfstop "train failed"; exit 1; }
R4=$(ls ckpt/r4-*.pth | sort | tail -1); cp "$R4" out/r4.pth; log "trained $R4"

(cd dl && python utils/export_dlshogi_onnx.py ../out/r4.pth ../out/r4.onnx) >> pipeline.log 2>&1
python - <<'PY'
import torch
c=torch.load('r3h.pth'); c['model']={k:(v.float() if v.is_floating_point() else v) for k,v in c['model'].items()}; torch.save(c,'r3.pth')
PY
(cd dl && python utils/make_opening_book.py ../r3.pth ../openings.txt --lines 200 --plies 12 --temperature 1.2 --top-k 8) >> pipeline.log 2>&1
log "match r4 vs r3 (800 playouts, SPRT 0/30, max 400 games)"
(cd dl && timeout 4h python -m pydlshogi2.match --engine1 /workspace/dl/scripts/usi_engine.sh --name1 r4 --options1 modelfile=../out/r4.pth,playouts=800 \
  --engine2 /workspace/dl/scripts/usi_engine.sh --name2 r3 --options2 modelfile=../r3.pth,playouts=800 \
  --games 400 --playouts 800 --opening ../openings.txt --sprt --elo0 0 --elo1 30 \
  --metrics ../out/match.jsonl --csa ../out/match-csa) > out/match.stdout 2>&1
tail -15 out/match.stdout >> pipeline.log
cp pipeline.log prep.log out/; touch out/DONE; log "DONE"
# Give the workstation 3 hours to fetch out/ and destroy the instance; otherwise stop ourselves.
for i in $(seq 1 180); do [ -f out/FETCHED ] && exit 0; sleep 60; done
selfstop "results not fetched in 3h (out/ kept on the stopped instance)"
