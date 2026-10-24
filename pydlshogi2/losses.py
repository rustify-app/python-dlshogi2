"""Loss functions used by policy-value training."""

import torch.nn.functional as F


def policy_cross_entropy(logits, move_labels, teacher_labels=None,
                         teacher_probabilities=None):
    """Compute hard-label or sparse soft-target policy cross entropy.

    Sparse teacher targets store only the engine's candidate move labels and
    their probabilities, avoiding a dense distribution over every legal move.
    """
    if teacher_labels is None or teacher_probabilities is None:
        return F.cross_entropy(logits, move_labels)
    if teacher_labels.shape != teacher_probabilities.shape:
        raise ValueError('teacher labels and probabilities must have the same shape')
    if teacher_labels.ndim != 2 or teacher_labels.shape[0] != logits.shape[0]:
        raise ValueError('teacher targets must have shape (batch, candidates)')
    log_probabilities = F.log_softmax(logits, dim=1)
    selected = log_probabilities.gather(1, teacher_labels.clamp_min(0))
    return -(selected * teacher_probabilities).sum(dim=1).mean()
