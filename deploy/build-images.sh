#!/usr/bin/env bash
# Build the floodgate and RL images, push them to zot, and set the tags in k8s/kustomization.yaml.
#   deploy/build-images.sh [floodgate|rl|all]      (default all)
# RL_REF: the branch with the native search and rl_loop.sh (default feat/native-search).
# Log in first: docker login zot.lovekapibarasan.org (ZOT_* in Infisical, Development
# Environment Variables / dev / general).
set -euo pipefail
cd "$(dirname "$0")/.."
REGISTRY="${REGISTRY:-zot.lovekapibarasan.org}"
RL_REF="${RL_REF:-feat/native-search}"
WHAT="${1:-all}"
K=deploy/k8s/kustomization.yaml

set_tag() {  # set_tag <image name in kustomization> <tag>
  python3 - "$K" "$1" "$2" <<'PY'
import re, sys
path, name, tag = sys.argv[1:]
s = open(path).read()
s = re.sub(rf'(- name: {re.escape(name)}\n\s+newName: [^\n]+\n\s+newTag: )"[^"]*"', rf'\g<1>"{tag}"', s)
open(path, "w").write(s)
PY
}

if [[ "$WHAT" == all || "$WHAT" == floodgate ]]; then
  TAG="$(git rev-parse --short HEAD)"
  docker build -f deploy/docker/floodgate.Dockerfile -t "$REGISTRY/dlshogi/floodgate:$TAG" .
  docker push "$REGISTRY/dlshogi/floodgate:$TAG"
  set_tag floodgate "$TAG"
fi

if [[ "$WHAT" == all || "$WHAT" == rl ]]; then
  git fetch -q origin "$RL_REF"
  WT="$(mktemp -d)"
  trap 'git worktree remove --force "$WT"' EXIT
  git worktree add -q --detach "$WT" "origin/$RL_REF"
  cp deploy/docker/rl-constraints.txt "$WT/deploy-constraints.txt"
  TAG="$(git -C "$WT" rev-parse --short HEAD)"
  docker build -f deploy/docker/rl.Dockerfile -t "$REGISTRY/dlshogi/rl:$TAG" "$WT"
  docker push "$REGISTRY/dlshogi/rl:$TAG"
  set_tag dlshogi-rl "$TAG"
fi
echo "commit $K and apply: kubectl --context karmada-apiserver apply -k deploy/k8s"
