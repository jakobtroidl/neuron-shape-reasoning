"""Saving and loading model weights.

A saved model is a directory with ``model.safetensors`` and ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .model import PATConfig, PointAffinityTransformer
from .presets import create_model

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

# Paper checkpoints, used by the research code (see research/README.md).
PAPER_CHECKPOINTS: dict[str, str] = {
    "flywire": "1MEbL3Yrci4ohQPPFPnvI9wDiq8b0wp5E",  # Google Drive file ids
    "hemibrain": "1jFRWQF265Znk9-KK2SQQmHNoP1dTLDHu",
    "manc": "1VvCxXsna-VZMTAtxotUBtRHlHs8ajyrX",
}


def save_pretrained(model: PointAffinityTransformer, path, *, dtype: torch.dtype | None = None, **metadata) -> Path:
    """Save weights and config to directory ``path``.

    Args:
        model: Model to save.
        path: Output directory, created if needed.
        dtype: Optionally cast floating-point weights (e.g. ``torch.float16``).
        **metadata: Extra JSON-serializable entries stored in ``config.json``.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        k: (v.to(dtype) if dtype is not None and v.is_floating_point() else v).contiguous()
        for k, v in model.state_dict().items()
    }
    save_file(state, str(path / WEIGHTS_NAME))
    (path / CONFIG_NAME).write_text(json.dumps({**model.config.to_dict(), **metadata}, indent=2))
    return path


def load_pretrained(path: str | os.PathLike, device="cpu") -> PointAffinityTransformer:
    """Load a saved model in eval mode.

    Args:
        path: A model directory, as written by `save_pretrained` or by a
            training run (``{output_dir}/model`` or ``{output_dir}/ckpt/epoch-N``).
        device: Device to move the model to.
    """
    from safetensors.torch import load_file

    path = Path(path)
    if not (path / WEIGHTS_NAME).is_file():
        raise FileNotFoundError(f"No model directory at {str(path)!r} (expected {WEIGHTS_NAME} and {CONFIG_NAME})")
    model = PointAffinityTransformer(PATConfig.from_dict(json.loads((path / CONFIG_NAME).read_text())))
    state = load_file(str(path / WEIGHTS_NAME), device="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).eval()


def load_legacy_checkpoint(path, *, preset: str | None = None, **overrides) -> PointAffinityTransformer:
    """Load a ``.pth`` checkpoint written by the original research code.

    The architecture is read from the checkpoint's stored arguments; ``preset``
    and ``overrides`` replace them. The research code divided coordinates (µm)
    by ``data_global_scale_factor`` (659.88367 for the paper models); apply the
    same scaling to your inputs.

    On torch >= 2.4 the file is loaded with ``weights_only=True``. Older
    versions unpickle it fully, so only load files you trust.
    """
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    else:  # torch < 2.4
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    args = vars(ckpt.get("args") or argparse.Namespace())
    model = create_model(preset or args.get("model", "ae_d1024_m512"), **{"depth": args.get("depth", 24), **overrides})
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval()


def download_paper_checkpoint(name: str, cache_dir=None) -> Path:
    """Download a paper checkpoint and convert it into a model directory.

    Needs ``gdown``. The model is stored in ``{cache_dir}/{name}``, by default
    ``$PAT_CACHE`` or ``~/.cache/pat``, and reused on later calls.

    Args:
        name: One of ``PAPER_CHECKPOINTS``.
        cache_dir: Where to store the converted model.

    Returns:
        The model directory, loadable with `load_pretrained`.
    """
    import gdown

    target = Path(cache_dir or os.environ.get("PAT_CACHE", Path.home() / ".cache" / "pat")) / name
    if (target / WEIGHTS_NAME).exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    legacy = target / "legacy.pth"
    print(f"Downloading {name} checkpoint (~4.5 GB, includes optimizer state) to {target} ...")
    gdown.download(id=PAPER_CHECKPOINTS[name], output=str(legacy), quiet=False)
    save_pretrained(load_legacy_checkpoint(legacy), target, source=f"{name} paper checkpoint")
    legacy.unlink()
    return target
