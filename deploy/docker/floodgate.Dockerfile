# syntax=docker/dockerfile:1.7
# floodgate player: FukauraOu (YaneuraOu's dlshogi-compatible engine, TensorRT) driven by
# pydlshogi2.csa_client. Built by .github/workflows/images.yml (context: main).
#
# This image only carries what we build: the engine binary, the few C++ runtime libraries it
# needs, and the client code (tens of MB). CUDA, TensorRT and Python come from elsewhere so
# nothing large goes through Zot (deploy/k8s/floodgate.yaml):
#   - the Pod runs on the public python image,
#   - an init container installs cshogi + the TensorRT / CUDA runtime wheels from PyPI into a
#     venv on the data volume (once per version set),
#   - another init container copies /dist from this image into the Pod.
ARG CUDA=12.9.1
# TensorRT 10.16.1.11 is what the engine ran with on ubuntu-03 (TRT101601 engine caches).
# The runtime side is the matching wheel tensorrt-cu12-libs==10.16.1.11.
ARG TRT=10.16.1.11-1+cuda12.9

FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS engine
ARG TRT
ARG YANEURAOU_REF=c1b80ea
RUN apt-get update && apt-get install -y --no-install-recommends git make clang lld ca-certificates \
      libnvinfer10=${TRT} libnvinfer-headers-dev=${TRT} libnvinfer-safe-headers-dev=${TRT} libnvinfer-dev=${TRT} \
      libnvonnxparsers10=${TRT} libnvonnxparsers-dev=${TRT} \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/yaneurao/YaneuraOu.git /src \
 && git -C /src checkout ${YANEURAOU_REF}
WORKDIR /src/source
# AVX2 matches the office nodes.
RUN make -j"$(nproc)" tournament YANEURAOU_EDITION=YANEURAOU_ENGINE_DEEP_TENSOR_RT_UBUNTU \
      TARGET_CPU=AVX2 COMPILER=clang++ \
 && install -D -m 0755 YaneuraOu-by-gcc /dist/engine/FukauraOu
# Bundle the engine's libraries the Pod's python image (Debian trixie) lacks, e.g. libgomp.
# glibc, libstdc++ and libgcc_s are already there (apt needs them); the NVIDIA ones come from
# the wheels and the driver.
RUN mkdir -p /dist/lib \
 && ldd /dist/engine/FukauraOu | awk '/=> \//{print $1, $3}' \
  | grep -vE '^(libc|libm|libdl|libpthread|librt|libstdc\+\+|libgcc_s|ld-linux[^ ]*)\.so|^lib(nvinfer|nvonnxparser|cudart|cuda|cublas)' \
  | while read -r name path; do cp -L "$path" "/dist/lib/$name"; done \
 && ls -la /dist/lib

FROM busybox:1.37
COPY --from=engine /dist /dist
COPY pydlshogi2 /dist/app/pydlshogi2
