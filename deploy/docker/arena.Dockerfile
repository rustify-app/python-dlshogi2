# Match arena (Issue #2): FukauraOu (TensorRT) + pydlshogi2.match on a single GPU host.
# Build context: engine.tgz (FukauraOu + lib/ from the floodgate image) and src (this repo).
#   docker run --gpus all -v $PWD/models:/models ... python -m pydlshogi2.match \
#     --options1 DNN_Model=/models/r3.onnx,NetworkDelay=0,NetworkDelay2=0,RoundUpToFullSecond=false ...
# /models must be writable: FukauraOu saves the compiled TensorRT engine next to the ONNX file and
# segfaults if it can't, which leaves the match waiting forever.
FROM python:3.14-slim-trixie
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cshogi==1.0.9 tensorrt-cu12-libs==10.16.1.11 nvidia-cuda-runtime-cu12==12.9.79
COPY engine.tgz /tmp/
RUN mkdir -p /opt/dlshogi && tar xzf /tmp/engine.tgz -C /opt/dlshogi && rm /tmp/engine.tgz
COPY src /opt/dlshogi/app
ENV PYTHONPATH=/opt/dlshogi/app PYTHONUNBUFFERED=1
RUN find / -xdev \( -name "libnvinfer.so.10*" -o -name "libnvonnxparser.so.10*" -o -name "libcudart.so.12*" \) -printf "%h\n" | sort -u > /etc/ld.so.conf.d/trt.conf && echo /opt/dlshogi/lib >> /etc/ld.so.conf.d/trt.conf && cat /etc/ld.so.conf.d/trt.conf && ldconfig && ! ldd /opt/dlshogi/engine/FukauraOu | grep "not found"
WORKDIR /opt/dlshogi/engine
