# floodgate player: FukauraOu (YaneuraOu's dlshogi-compatible engine, TensorRT) driven by
# pydlshogi2.csa_client. Built by .github/workflows/images.yml (context: main).
ARG CUDA=12.9.1
# TensorRT 10.16.1.11 is what the engine ran with on ubuntu-03 (TRT101601 engine caches).
ARG TRT=10.16.1.11-1+cuda12.9

FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS engine
ARG TRT
ARG YANEURAOU_REF=c1b80ea
RUN apt-get update && apt-get install -y --no-install-recommends git make clang ca-certificates \
      libnvinfer10=${TRT} libnvinfer-headers-dev=${TRT} libnvinfer-safe-headers-dev=${TRT} libnvinfer-dev=${TRT} \
      libnvonnxparsers10=${TRT} libnvonnxparsers-dev=${TRT} \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/yaneurao/YaneuraOu.git /src \
 && git -C /src checkout ${YANEURAOU_REF}
WORKDIR /src/source
# AVX2 matches the office nodes (Xeon E5 v4).
RUN make -j"$(nproc)" tournament YANEURAOU_EDITION=YANEURAOU_ENGINE_DEEP_TENSOR_RT_UBUNTU \
      TARGET_CPU=AVX2 COMPILER=clang++ \
 && install -m 0755 YaneuraOu-by-gcc /FukauraOu

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
ARG TRT
RUN apt-get update && apt-get install -y --no-install-recommends \
      libnvinfer10=${TRT} libnvonnxparsers10=${TRT} libgomp1 python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir cshogi==1.0.9
COPY --from=engine /FukauraOu /engine/FukauraOu
# The models and TensorRT engine caches live on the volume (FukauraOu reads ./eval/<DNN_Model>).
RUN ln -s /data/eval /engine/eval
COPY pydlshogi2 /app/pydlshogi2
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app PYTHONUNBUFFERED=1
WORKDIR /app
ENTRYPOINT ["python", "-m", "pydlshogi2.csa_client", "--engine", "/engine/FukauraOu", "--engine-dir", "/engine"]
