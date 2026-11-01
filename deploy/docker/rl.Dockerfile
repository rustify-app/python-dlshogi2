# Self-play RL loop (rl_loop.sh) with the native UCT core and the TensorRT backend.
# Build context: a checkout of the branch that has the native search (feat/native-search);
# .github/workflows/images.yml checks that branch out as the context. Versions are pinned to what ran on ubuntu-03.
FROM python:3.14-slim-trixie AS native
RUN apt-get update && apt-get install -y --no-install-recommends g++ git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG CSHOGI_VERSION=1.0.9
RUN git clone --depth 1 --branch v${CSHOGI_VERSION} https://github.com/TadaoYamaoka/cshogi.git /cshogi-src
COPY pydlshogi2/uct/native /native
# -march=broadwell instead of build.sh's -march=native: built here, run on the office Xeon E5 v4.
RUN cd /native && sed -i 's/-march=native/-march=broadwell/' build.sh && CSHOGI_SRC=/cshogi-src sh build.sh

FROM python:3.14-slim-trixie
COPY deploy-constraints.txt /tmp/constraints.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt \
 && pip uninstall -y onnxruntime \
 && pip install --no-cache-dir -c /tmp/constraints.txt onnxruntime-gpu tensorrt_cu13
WORKDIR /app
COPY . /app
COPY --from=native /native/libuct.so /app/pydlshogi2/uct/native/libuct.so
RUN pip install --no-cache-dir --no-deps -e /app
# State is on the volume: WORKDIR (rl1), checkpoints, openings, ~/.cache (TensorRT engines).
ENV HOME=/data/home PYTHONUNBUFFERED=1 PYTHON=python WORKDIR=/data/rl1 GATE_OPENING=/data/openings.txt
ENTRYPOINT ["bash", "/app/rl_loop.sh"]
CMD ["/data/checkpoints/checkpoint.pth"]
