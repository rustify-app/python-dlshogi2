"""ONNX inference backend for the MCTS engine.

:class:`OnnxPlayer` subclasses :class:`~pydlshogi2.player.mcts_player.MCTSPlayer`
and swaps only the model loading, feature buffer and inference steps so the
network runs through :mod:`onnxruntime` instead of PyTorch. Everything else
(search, time control, USI handling) is inherited unchanged.

The player accepts two model layouts and picks one from the ONNX input names:

* a single ``input`` (:mod:`utils.export_onnx`, the pydlshogi2 feature set)
* ``input1``/``input2`` (:mod:`utils.export_dlshogi_onnx`, dlshogi WCSC28 planes)

Features and move labels then match the ones the model was trained with.
"""
import numpy as np
import onnxruntime

from pydlshogi2.features import (
    feature_channels, input_feature_function, move_label_function,
)
from pydlshogi2.player.mcts_player import MCTSPlayer


class OnnxPlayer(MCTSPlayer):
    """MCTS player that performs network inference with onnxruntime."""

    #: USI engine name reported to the GUI.
    name = "python-dlshogi-onnx"
    #: Default model path used when ``modelfile`` is not overridden via USI.
    DEFAULT_MODELFILE = "model/model.onnx"

    def __init__(self):
        super().__init__()
        #: Whether to enable the TensorRT execution provider (fastest, requires
        #: a TensorRT-enabled onnxruntime build).
        self.tensorrt = False

    def usi(self):
        """Print USI options, adding the ONNX-specific ``tensorrt`` toggle."""
        super().usi()
        print("option name tensorrt type check default false")

    def setoption(self, args):
        """Handle the ``tensorrt`` option, delegating the rest to the base."""
        if args[1] == "tensorrt":
            self.tensorrt = args[3] == "true"
        else:
            super().setoption(args)

    def load_model(self):
        """Create the onnxruntime inference session."""
        if self.gpu_id >= 0:
            providers = []
            if self.tensorrt:
                providers.append(("TensorrtExecutionProvider", {"device_id": self.gpu_id}))
            providers.append(("CUDAExecutionProvider", {"device_id": self.gpu_id}))
            providers.append("CPUExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(self.modelfile, providers=providers)
        input_names = [i.name for i in self.session.get_inputs()]
        self.feature_set = 'dlshogi' if 'input1' in input_names else 'pydlshogi2'
        self.make_features = input_feature_function(self.feature_set)
        self.make_label = move_label_function(self.feature_set)
        if self.feature_set == 'dlshogi':
            self.split = self.session.get_inputs()[0].shape[1]

    def init_features(self):
        """Allocate the feature buffer for the model's feature set."""
        channels = feature_channels(self.feature_set)
        self.features = np.empty((self.batch_size, channels, 9, 9), dtype=np.float32)

    def make_input_features(self, board):
        """Write ``board`` into the current batch slot of the feature buffer."""
        self.make_features(board, self.features[self.current_batch_index])

    def infer(self):
        """Run inference over the queued positions."""
        x = self.features[0:self.current_batch_index]
        if self.feature_set == 'dlshogi':
            feeds = {"input1": np.ascontiguousarray(x[:, :self.split]),
                     "input2": np.ascontiguousarray(x[:, self.split:])}
        else:
            feeds = {"input": x}
        policy, value = self.session.run(["output_policy", "output_value"], feeds)
        return policy, value


if __name__ == "__main__":
    player = OnnxPlayer()
    player.run()
