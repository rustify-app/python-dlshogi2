"""TensorRT inference for the native search.

The native search is GPU-bound: with a 10x192 network at batch 32 the RTX 2060
spends most of the search evaluating positions, and PyTorch — even replaying a
CUDA graph in fp16 — leaves speed on the table that TensorRT's kernel selection
and layer fusion pick up.

:class:`TrtEvaluator` builds (once) and runs a TensorRT engine for a fixed batch
size.  The engine is cached next to nothing the user has to manage: under
``~/.cache/pydlshogi2/trt``, keyed by the checkpoint's contents, the batch size
and the TensorRT version, so a new checkpoint or a TensorRT upgrade rebuilds it
and nothing else does.  Building takes a minute or two.

It exposes the same launch/wait interface the native search pipeline uses for
its CUDA graphs: one execution context and one set of device buffers per slot,
outputs copied into pinned host arrays the C++ side reads directly.
"""
import hashlib
import os
import tempfile

import numpy as np
import torch

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pydlshogi2', 'trt')


def _checkpoint_digest(modelfile):
    h = hashlib.sha1()
    with open(modelfile, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def engine_path(modelfile, batch_size):
    import tensorrt as trt
    name = '{}-b{}-fp16s-trt{}.engine'.format(_checkpoint_digest(modelfile), batch_size, trt.__version__)
    return os.path.join(CACHE_DIR, name)


def _export_onnx(modelfile, onnxfile):
    """Export an fp16 graph with fp32 inputs and outputs.

    TensorRT 11 dropped the builder's FP16 flag: the precision comes from the
    network itself ("strongly typed").  So the graph casts its float32 input to
    float16, runs the (BatchNorm-folded) network in float16, and casts the
    outputs back — the interface stays float32.
    """
    from pydlshogi2.features import FEATURES_NUM
    from pydlshogi2.network.policy_value_resnet import load_network, fuse_for_inference

    class InferenceModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            policy, value = self.model(x.half())
            return policy.float(), torch.sigmoid(value.float())

    device = torch.device('cuda')
    model, _ = load_network(modelfile, device)
    model = fuse_for_inference(model.eval()).half()
    wrapper = InferenceModel(model).eval()
    dummy = torch.zeros((1, FEATURES_NUM, 9, 9), dtype=torch.float32, device=device)
    torch.onnx.export(wrapper, dummy, onnxfile, input_names=['input'],
                      output_names=['output_policy', 'output_value'],
                      dynamic_axes={'input': {0: 'batch'}, 'output_policy': {0: 'batch'},
                                    'output_value': {0: 'batch'}},
                      opset_version=17, dynamo=False)


def build_engine(modelfile, batch_size, path, log=print):
    """Export ``modelfile`` to ONNX and build an fp16 TensorRT engine for ``batch_size``."""
    import tensorrt as trt
    from pydlshogi2.features import FEATURES_NUM

    logger = trt.Logger(trt.Logger.WARNING)
    with tempfile.TemporaryDirectory() as tmp:
        onnxfile = os.path.join(tmp, 'model.onnx')
        _export_onnx(modelfile, onnxfile)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        with open(onnxfile, 'rb') as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError('ONNX parse failed: ' + '; '.join(errors))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        profile = builder.create_optimization_profile()
        shape = (batch_size, FEATURES_NUM, 9, 9)
        profile.set_shape('input', shape, shape, shape)
        config.add_optimization_profile(profile)
        log('building TensorRT engine (batch {}) for {} ...'.format(batch_size, modelfile))
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT engine build failed')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'wb') as f:
        f.write(serialized)
    os.replace(path + '.tmp', path)
    _prune_cache(keep=8)


def _prune_cache(keep):
    """Keep only the ``keep`` most recently used engines (RL builds one per checkpoint)."""
    engines = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.engine')]
    engines.sort(key=os.path.getmtime, reverse=True)
    for stale in engines[keep:]:
        try:
            os.remove(stale)
        except OSError:
            pass


class TrtEvaluator:
    """Evaluate fixed-size batches with a TensorRT engine, one context per slot.

    :param modelfile: checkpoint to evaluate (the engine is built and cached on first use).
    :param batch_size: positions per batch.
    :param nslots: number of pipeline slots (each has its own buffers and context).
    :param features: pinned ``float32`` tensor of ``nslots * batch_size`` input positions.
    :param device: the CUDA device.
    """

    def __init__(self, modelfile, batch_size, nslots, features, device, log=print):
        import tensorrt as trt
        path = engine_path(modelfile, batch_size)
        if not os.path.exists(path):
            build_engine(modelfile, batch_size, path, log)
        else:
            os.utime(path)  # 使ったものほど消されにくくする
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.batch_size = batch_size
        self.features = features
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        B = batch_size
        shape = (B,) + tuple(features.shape[1:])
        n_labels = self.engine.get_tensor_shape('output_policy')[1]
        self.contexts = []
        self.buffers = []
        for _ in range(nslots):
            ctx = self.engine.create_execution_context()
            ctx.set_input_shape('input', shape)
            x = torch.empty(shape, dtype=torch.float32, device=device)
            p = torch.empty((B, n_labels), dtype=torch.float32, device=device)
            v = torch.empty((B, 1), dtype=torch.float32, device=device)
            ctx.set_tensor_address('input', x.data_ptr())
            ctx.set_tensor_address('output_policy', p.data_ptr())
            ctx.set_tensor_address('output_value', v.data_ptr())
            self.contexts.append(ctx)
            self.buffers.append((x, p, v))
        self.policy_pinned = torch.empty((nslots * B, n_labels), dtype=torch.float32, pin_memory=True)
        self.value_pinned = torch.empty(nslots * B, dtype=torch.float32, pin_memory=True)
        self.policy_out = self.policy_pinned.numpy()
        self.value_out = self.value_pinned.numpy()
        self.events = [torch.cuda.Event() for _ in range(nslots)]

    def launch(self, slot, n):
        B = self.batch_size
        x, p, v = self.buffers[slot]
        with torch.cuda.stream(self.stream):
            # バッチが満杯でなくても全体を流す (後ろの行は前回の残りで、結果は読まれない)
            x.copy_(self.features[slot * B:(slot + 1) * B], non_blocking=True)
            self.contexts[slot].execute_async_v3(self.stream.cuda_stream)
            self.policy_pinned[slot * B:(slot + 1) * B].copy_(p, non_blocking=True)
            self.value_pinned[slot * B:(slot + 1) * B].copy_(v.view(-1), non_blocking=True)
            self.events[slot].record(self.stream)
        return 0

    def wait(self, slot):
        self.events[slot].synchronize()
        return 0
