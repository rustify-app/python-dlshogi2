# syntax=docker/dockerfile:1.7
# Self-play RL loop (rl_loop.sh) with the native UCT core and the TensorRT backend.
# Build context: a checkout of the branch that has the native search (feat/native-search);
# .github/workflows/images.yml checks that branch out as the context.
#
# This image only carries the code and the native core (a few MB). torch, onnxruntime-gpu and
# TensorRT are installed from PyPI by an init container into a venv on the data volume, pinned
# by deploy-constraints.txt (what ran on ubuntu-03), so nothing large goes through Zot
# (deploy/k8s/rl1.yaml). The Pod itself runs on the public python image.
FROM python:3.14-slim-trixie AS native
RUN apt-get update && apt-get install -y --no-install-recommends g++ git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG CSHOGI_VERSION=1.0.9
RUN git clone --depth 1 --branch v${CSHOGI_VERSION} https://github.com/TadaoYamaoka/cshogi.git /cshogi-src
COPY pydlshogi2/uct/native /native
# -march=broadwell instead of build.sh's -march=native: built here, run on the office nodes.
RUN cd /native && sed -i 's/-march=native/-march=broadwell/' build.sh && CSHOGI_SRC=/cshogi-src sh build.sh

FROM busybox:1.37
COPY --exclude=.git . /dist/app
COPY --from=native /native/libuct.so /dist/app/pydlshogi2/uct/native/libuct.so
