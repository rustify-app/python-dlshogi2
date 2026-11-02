"""Rebuild a training checkpoint from a dlshogi-style ONNX model (the inverse of export_dlshogi_onnx.py).

The r3 .pth was lost with its Vast.ai instance; only r3.onnx survived. The exporter folds each
BatchNorm into the Conv before it, so those convs come back as conv weight = folded weight and an
identity BatchNorm carrying the folded bias (running_mean 0, running_var 1 - eps). Everything else
(SE, policy conv/bias, value FCs) keeps its PyTorch name in the ONNX file and is copied by name.

    python utils/onnx_to_checkpoint.py r3.onnx r3.pth [--check]

--check runs both models on random inputs (needs onnxruntime) and prints the largest difference.
The result trains with ``pydlshogi2.train --resume r3.pth`` (epoch/step start at 0, fresh SGD state).
"""
import argparse
import os
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2.network.policy_value_resnet import build_network  # noqa: E402


def convert(onnxfile):
    m = onnx.load(onnxfile)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    named = {k[len('model.'):]: v for k, v in inits.items() if k.startswith('model.')}

    # Folded convs in graph (= forward) order: conv1, blocks.*.conv1/conv2, value_conv1.
    folded = []
    for node in m.graph.node:
        if node.op_type == 'Conv' and node.input[1] in inits and not node.input[1].startswith('model.'):
            folded.append((inits[node.input[1]], inits[node.input[2]]))

    blocks = sum(1 for k in named if k.endswith('.se.fc1.weight')) or len(folded) // 2 - 1
    channels = folded[0][0].shape[0]
    config = {'blocks': blocks, 'channels': channels, 'fcl': named['value_fc1.weight'].shape[0],
              'se': any('.se.' in k for k in named), 'features': 'dlshogi'}
    model = build_network(config)
    targets = [('conv1', 'norm1')]
    for i in range(blocks):
        targets += [('blocks.{}.conv1'.format(i), 'blocks.{}.bn1'.format(i)),
                    ('blocks.{}.conv2'.format(i), 'blocks.{}.bn2'.format(i))]
    targets.append(('value_conv1', 'value_norm1'))
    if len(targets) != len(folded):
        raise SystemExit('expected {} folded convs, found {}'.format(len(targets), len(folded)))

    sd = model.state_dict()
    for (conv, bn), (w, b) in zip(targets, folded):
        eps = dict(model.named_modules())[bn].eps
        sd[conv + '.weight'] = torch.from_numpy(w.copy())
        sd[bn + '.weight'] = torch.ones(len(b))
        sd[bn + '.bias'] = torch.from_numpy(b.copy())
        sd[bn + '.running_mean'] = torch.zeros(len(b))
        sd[bn + '.running_var'] = torch.full((len(b),), 1.0 - eps)
    for k, v in named.items():
        if k not in sd or tuple(sd[k].shape) != v.shape:
            raise SystemExit('unexpected initializer {} {}'.format(k, v.shape))
        sd[k] = torch.from_numpy(v.copy())
    model.load_state_dict(sd)
    return model, config


def check(model, onnxfile):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnxfile, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    x1 = (rng.random((8, 62, 9, 9)) < 0.1).astype(np.float32)
    x2 = (rng.random((8, 57, 9, 9)) < 0.1).astype(np.float32)
    p_onnx, v_onnx = sess.run(None, {'input1': x1, 'input2': x2})
    model.eval()
    with torch.no_grad():
        p, v = model(torch.cat((torch.from_numpy(x1), torch.from_numpy(x2)), 1))
    dp = np.abs(p.numpy() - p_onnx).max()
    dv = np.abs(torch.sigmoid(v).numpy() - v_onnx).max()
    print('max |policy diff| {:.2e}, max |value diff| {:.2e}'.format(dp, dv))
    return dp < 1e-3 and dv < 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('onnxfile')
    parser.add_argument('checkpoint')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    model, config = convert(args.onnxfile)
    if args.check and not check(model, args.onnxfile):
        raise SystemExit('outputs differ from the ONNX model')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    torch.save({'epoch': 0, 't': 0, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'network': config}, args.checkpoint)
    print('{} -> {} ({})'.format(args.onnxfile, args.checkpoint, config))


if __name__ == '__main__':
    main()
