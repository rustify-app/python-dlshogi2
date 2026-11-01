#!/usr/bin/env bash
# Point deploy/k8s at the images CI built (.github/workflows/images.yml tags them with the full
# commit sha): floodgate from origin/main, rl from origin/feat/native-search.
#   deploy/set-image-tags.sh [floodgate-sha] [rl-sha]
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch -q origin main feat/native-search
FG="${1:-$(git rev-parse origin/main)}"
RL="${2:-$(git rev-parse origin/feat/native-search)}"
python3 - deploy/k8s/kustomization.yaml "$FG" "$RL" <<'PY'
import re, sys
path, fg, rl = sys.argv[1:]
s = open(path).read()
for name, tag in (("floodgate", fg), ("dlshogi-rl", rl)):
    s = re.sub(rf'(- name: {name}\n\s+newName: [^\n]+\n\s+newTag: )"[^"]*"', rf'\g<1>"{tag}"', s)
open(path, "w").write(s)
PY
grep -n newTag deploy/k8s/kustomization.yaml
