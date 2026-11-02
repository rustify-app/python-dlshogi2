#!/bin/bash
# floodgate archives -> hcpe (per year, two rating filters). Logs in /workspace/prep.log
set -u
cd /workspace
pip install -q cshogi numpy scikit-learn onnx onnxscript onnxruntime >/dev/null 2>&1; echo pip=$?
command -v 7z >/dev/null || (apt-get update -qq && apt-get install -y -qq p7zip-full >/dev/null 2>&1); echo 7z=$(command -v 7z)
mkdir -p arc csa hcpe
for y in 2019 2020 2021 2022 2023 2024 2025 2026; do
  ( curl -s -o arc/wdoor$y.7z https://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoor$y.7z && 7z x -y -bd -oCSA_$y arc/wdoor$y.7z >/dev/null && mv CSA_$y csa/$y && rm arc/wdoor$y.7z && echo "extracted $y $(find csa/$y -name "*.csa" | wc -l) files"
    for r in 3800 3500; do python dl/utils/csa_to_hcpe.py csa/$y hcpe/$y-$r-train.hcpe hcpe/$y-$r-test.hcpe --filter_rating $r --test_ratio 0.02 > hcpe/$y-$r.log 2>&1 & done; wait
    echo "converted $y: $(grep -h position_num hcpe/$y-3800.log | tr "\n" " ") | 3500: $(grep -h position_num hcpe/$y-3500.log | tr "\n" " ")"
    rm -rf csa/$y ) &
done
wait; echo ALL DONE
