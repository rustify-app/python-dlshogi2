import os
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


def save_checkpoint(path, features):
    from pydlshogi2.network.policy_value_resnet import build_network
    config = {'blocks': 1, 'channels': 16, 'fcl': 16, 'se': False, 'features': features}
    model = build_network(config)
    torch.save({'model': model.state_dict(), 'network': config, 'epoch': 0, 't': 0}, path)


def play(onnxfile):
    commands = '\n'.join([
        'usi',
        'setoption name modelfile value ' + onnxfile,
        'setoption name gpu_id value -1',
        'setoption name playouts value 30',
        'isready',
        'position startpos moves 7g7f 3c3d',
        'go',
        'quit',
    ]) + '\n'
    result = subprocess.run(
        [sys.executable, '-m', 'pydlshogi2.player.onnx_player'], input=commands,
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    return result.stdout + result.stderr


@unittest.skipUnless(HAVE_ORT, 'onnxruntime is not installed')
class OnnxPlayerFormatTest(unittest.TestCase):
    def check(self, features, exporter):
        with tempfile.TemporaryDirectory() as tmp:
            pth = os.path.join(tmp, 'model.pth')
            onnxfile = os.path.join(tmp, 'model.onnx')
            save_checkpoint(pth, features)
            subprocess.run([sys.executable, os.path.join(ROOT, 'utils', exporter), pth, onnxfile],
                           check=True, capture_output=True, cwd=ROOT, timeout=300,
                           env=dict(os.environ, PYTHONPATH=ROOT))
            output = play(onnxfile)
        self.assertIn('bestmove ', output, output[-2000:])
        self.assertNotIn('Traceback', output, output[-2000:])

    def test_single_input_model(self):
        self.check('pydlshogi2', 'export_onnx.py')

    def test_dlshogi_two_input_model(self):
        self.check('dlshogi', 'export_dlshogi_onnx.py')


if __name__ == '__main__':
    unittest.main()
