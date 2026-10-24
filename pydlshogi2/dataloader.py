import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import torch

from cshogi import Board, HuffmanCodedPosAndEval
from pydlshogi2.features import (
    FEATURES_NUM, MOVE_LABELS_NUM, DEFAULT_EVAL_COEF,
    make_input_features, make_move_label, make_value_target,
)


class HcpeDataLoader:
    """Mini-batch loader for HCPE (HuffmanCodedPosAndEval) training data.

    :param files: one path or a list of paths to ``.hcpe`` files.
    :param batch_size: number of positions per mini-batch.
    :param device: torch device the batches are moved to.
    :param shuffle: whether to shuffle the data each epoch.
    :param val_lambda: weight on the game outcome for the value target; ``1.0``
        uses the outcome only (original behaviour), values below ``1.0`` blend
        in the stored search evaluation (see :func:`pydlshogi2.features.make_value_target`).
    :param eval_coef: sigmoid temperature for the evaluation-to-win-rate mapping.
    """

    def __init__(self, files, batch_size, device, shuffle=False,
                 val_lambda=1.0, eval_coef=DEFAULT_EVAL_COEF,
                 teacher_targets=None, deduplicate_positions=False,
                 recency_decay=1.0):
        if not 0 < recency_decay <= 1:
            raise ValueError('recency_decay must be greater than 0 and at most 1')
        self.deduplicate_positions = deduplicate_positions
        self.recency_decay = recency_decay
        self.load(files)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.val_lambda = val_lambda
        self.eval_coef = eval_coef
        self.teacher_targets = self._load_teacher_targets(teacher_targets)
        self.teacher_width = (len(next(iter(self.teacher_targets.values()))[0])
                              if self.teacher_targets else 1)

        # pinned memory はCUDAのみ利用可能
        pin = device.type == 'cuda'
        self.torch_features = torch.empty((batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=pin)
        self.torch_move_label = torch.empty((batch_size), dtype=torch.int64, pin_memory=pin)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=pin)
        if self.teacher_targets is not None:
            self.torch_teacher_labels = torch.zeros(
                (batch_size, self.teacher_width), dtype=torch.int64, pin_memory=pin)
            self.torch_teacher_probs = torch.zeros(
                (batch_size, self.teacher_width), dtype=torch.float32, pin_memory=pin)

        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        if self.teacher_targets is not None:
            self.teacher_labels = self.torch_teacher_labels.numpy()
            self.teacher_probs = self.torch_teacher_probs.numpy()

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.board = Board()

    def load(self, files):
        data = []
        source_indices = []
        if type(files) not in [list, tuple]:
            files = [files]
        source_index = 0
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                chunk = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
                data.append(chunk)
                source_indices.append(np.full(len(chunk), source_index, dtype=np.int32))
                source_index += 1
            else:
                logging.warn('{} not found, skipping'.format(path))
        self.data = np.concatenate(data)
        self.source_indices = np.concatenate(source_indices)
        self.duplicate_positions_removed = 0
        if self.deduplicate_positions:
            hcp_view = self.data['hcp'].reshape(len(self.data), -1)
            # Traverse backwards so the newest file supplies the surviving record.
            _, reverse_first = np.unique(hcp_view[::-1], axis=0, return_index=True)
            keep = np.sort(len(self.data) - 1 - reverse_first)
            self.duplicate_positions_removed = len(self.data) - len(keep)
            self.data = self.data[keep]
            self.source_indices = self.source_indices[keep]

        if (self.recency_decay == 1.0 or not len(self.source_indices)
                or self.source_indices.min() == self.source_indices.max()):
            self.sampling_probabilities = None
        else:
            age = self.source_indices.max() - self.source_indices
            weights = np.power(self.recency_decay, age).astype(np.float64)
            self.sampling_probabilities = weights / weights.sum()

    @staticmethod
    def _load_teacher_targets(path):
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as archive:
            required = {'hcp', 'labels', 'probabilities'}
            if not required.issubset(archive.files):
                raise ValueError('teacher target archive is missing required arrays')
            hcps = np.asarray(archive['hcp'], dtype=np.uint8)
            labels = np.asarray(archive['labels'], dtype=np.int64)
            probabilities = np.asarray(archive['probabilities'], dtype=np.float32)

        if hcps.ndim != 2 or labels.ndim != 2 or probabilities.shape != labels.shape:
            raise ValueError('teacher target arrays have invalid shapes')
        if hcps.shape[0] != labels.shape[0] or labels.shape[1] < 1:
            raise ValueError('teacher target arrays have inconsistent lengths')
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError('teacher probabilities must be finite and non-negative')
        valid = probabilities > 0
        if np.any(labels[valid] < 0) or np.any(labels[valid] >= MOVE_LABELS_NUM):
            raise ValueError('teacher move label is outside the policy head')
        if np.any(np.abs(probabilities.sum(axis=1) - 1.0) > 1e-4):
            raise ValueError('each teacher probability row must sum to one')

        return {
            bytes(hcp): (row_labels, row_probs)
            for hcp, row_labels, row_probs in zip(hcps, labels, probabilities)
        }

    def mini_batch(self, hcpevec):
        self.features.fill(0)
        if self.teacher_targets is not None:
            self.teacher_labels.fill(0)
            self.teacher_probs.fill(0)
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])
            make_input_features(self.board, self.features[i])
            label = make_move_label(hcpe['bestMove16'], self.board.turn)
            self.move_label[i] = label
            if self.teacher_targets is not None:
                target = self.teacher_targets.get(bytes(hcpe['hcp']))
                if target is None:
                    self.teacher_labels[i, 0] = label
                    self.teacher_probs[i, 0] = 1.0
                else:
                    target_labels, target_probs = target
                    self.teacher_labels[i] = target_labels
                    self.teacher_probs[i] = target_probs
            self.result[i] = make_value_target(
                hcpe['gameResult'], hcpe['eval'], self.board.turn,
                self.val_lambda, self.eval_coef)

        if self.device.type == 'cpu':
            batch = (self.torch_features.clone(),
                     self.torch_move_label.clone(),
                     self.torch_result.clone())
            if self.teacher_targets is not None:
                return batch + (self.torch_teacher_labels.clone(),
                                self.torch_teacher_probs.clone())
            return batch

        batch = (self.torch_features.to(self.device),
                 self.torch_move_label.to(self.device),
                 self.torch_result.to(self.device))
        if self.teacher_targets is not None:
            return batch + (self.torch_teacher_labels.to(self.device),
                            self.torch_teacher_probs.to(self.device))
        return batch

    def sample(self):
        indices = np.random.choice(
            len(self.data), self.batch_size, replace=False,
            p=self.sampling_probabilities)
        return self.mini_batch(self.data[indices])

    def pre_fetch(self):
        hcpevec = self.epoch_data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.i = 0
        if self.sampling_probabilities is not None:
            indices = np.random.choice(
                len(self.data), len(self.data), replace=True,
                p=self.sampling_probabilities)
            self.epoch_data = self.data[indices]
        else:
            self.epoch_data = self.data.copy()
            if self.shuffle:
                np.random.shuffle(self.epoch_data)
        self.pre_fetch()
        return self

    def __next__(self):
        if self.i > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result
