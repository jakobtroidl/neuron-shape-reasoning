"""Training checkpoints (model + optimizer state), loadable with ``weights_only=True``."""

from __future__ import annotations

from pathlib import Path

import torch

from ..model import PATConfig, PointAffinityTransformer


def save_checkpoint(path, model: PointAffinityTransformer, optimizer, epoch: int, train_config: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "train_config": train_config,
        },
        path,
    )
    return path


def load_checkpoint(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(path) -> PointAffinityTransformer:
    ckpt = load_checkpoint(path)
    model = PointAffinityTransformer(PATConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()
