"""Training checkpoints: a model directory plus the optimizer state.

A checkpoint directory loads with [`load_pretrained`][pat.load_pretrained] like
any saved model, and resumes training with ``resume: <directory>``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..model import PointAffinityTransformer
from ..pretrained import save_pretrained

STATE_NAME = "training_state.pt"


def save_checkpoint(path, model: PointAffinityTransformer, optimizer, epoch: int, train_config: dict) -> Path:
    """Write the model (`save_pretrained` format) and ``training_state.pt`` to directory ``path``."""
    path = save_pretrained(model, path, epoch=epoch)
    torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch, "train_config": train_config}, path / STATE_NAME)
    return path


def load_training_state(path) -> dict:
    """Optimizer state, epoch and training config of a checkpoint directory."""
    return torch.load(Path(path) / STATE_NAME, map_location="cpu", weights_only=True)
