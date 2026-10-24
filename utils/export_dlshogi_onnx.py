"""Export a checkpoint trained with ``--features dlshogi`` to a dlshogi-style ONNX model.

The graph follows the dlshogi (WCSC28) inference contract, which the dlshogi
C++ engine and YaneuraOu's FukauraOu edition load directly:

* inputs  ``input1``: ``float32[batch, 62, 9, 9]`` and ``input2``: ``float32[batch, 57, 9, 9]``
  (the planes produced by ``cshogi.dlshogi.make_input_features``)
* outputs ``output_policy``: ``float32[batch, 2187]`` policy logits indexed by
  ``cshogi.dlshogi.make_move_label``, and ``output_value``: ``float32[batch, 1]``
  win probability of the side to move (sigmoid already applied)

Example:
    python utils/export_dlshogi_onnx.py checkpoints/model.pth model.onnx
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2.network.policy_value_resnet import load_network  # noqa: E402


class DlshogiInferenceModel(nn.Module):
    """Two-input wrapper concatenating the dlshogi planes for the backbone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input1, input2):
        policy, value = self.model(torch.cat((input1, input2), dim=1))
        return policy, torch.sigmoid(value)


def export(modelfile, onnxfile, device='cpu', opset=17):
    import cshogi.dlshogi as dlshogi

    model, _ = load_network(modelfile, device)
    if model.config.get('features') != 'dlshogi':
        raise SystemExit('{} was not trained with --features dlshogi'.format(modelfile))
    model.eval()
    wrapper = DlshogiInferenceModel(model).eval()

    input1 = torch.zeros((1, dlshogi.FEATURES1_NUM, 9, 9), dtype=torch.float32, device=device)
    input2 = torch.zeros((1, dlshogi.FEATURES2_NUM, 9, 9), dtype=torch.float32, device=device)
    torch.onnx.export(
        wrapper, (input1, input2), onnxfile,
        input_names=['input1', 'input2'],
        output_names=['output_policy', 'output_value'],
        dynamic_axes={name: {0: 'batch'} for name in
                      ('input1', 'input2', 'output_policy', 'output_value')},
        opset_version=opset,
        dynamo=False,
    )
    print('Exported {} -> {}'.format(modelfile, onnxfile))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modelfile')
    parser.add_argument('onnxfile')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    export(args.modelfile, args.onnxfile, args.device, args.opset)


if __name__ == '__main__':
    main()
