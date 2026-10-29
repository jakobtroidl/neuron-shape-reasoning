"""Saving and loading model weights.

A saved model is a directory with ``model.safetensors`` and ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import torch

from .model import PATConfig, PointAffinityTransformer
from .presets import PRESETS, preset_config

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

# Paper checkpoints, used by the research code (see research/README.md). Downloaded from
# Google Drive and converted on first use.
PRETRAINED: dict[str, dict] = {
    "flywire": {"gdrive_id": "1MEbL3Yrci4ohQPPFPnvI9wDiq8b0wp5E"},
    "hemibrain": {"gdrive_id": "1jFRWQF265Znk9-KK2SQQmHNoP1dTLDHu"},
    "manc": {"gdrive_id": "1VvCxXsna-VZMTAtxotUBtRHlHs8ajyrX"},
}


def cache_dir() -> Path:
    """Directory for downloaded weights: ``$PAT_CACHE`` or ``~/.cache/pat``."""
    return Path(os.environ.get("PAT_CACHE", Path.home() / ".cache" / "pat"))


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


def _load_dir(path: Path, device) -> PointAffinityTransformer:
    from safetensors.torch import load_file

    config = PATConfig.from_dict(json.loads((path / CONFIG_NAME).read_text()))
    model = PointAffinityTransformer(config)
    state = load_file(str(path / WEIGHTS_NAME), device="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).eval()


def load_legacy_checkpoint(path, *, preset: str | None = None, **overrides) -> PointAffinityTransformer:
    """Load a ``.pth`` checkpoint written by the original research code.

    The architecture is read from the checkpoint's stored arguments unless
    given explicitly. The research code divided coordinates (µm) by
    ``data_global_scale_factor`` (659.88367 for the paper models); apply the
    same scaling to your inputs. Loading uses ``weights_only=True``.
    """
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    else:  # torch < 2.4
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    args = vars(ckpt.get("args") or argparse.Namespace())
    preset = preset or args.get("model", "ae_d1024_m512")
    if preset not in PRESETS:
        raise KeyError(f"Checkpoint uses unknown model {preset!r}")
    fields = {"depth": args.get("depth", 24)}
    config = preset_config(preset, **{**fields, **overrides})

    model = PointAffinityTransformer(config)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval()


def convert_legacy_checkpoint(src, dst, *, dtype: torch.dtype | None = None, **kwargs) -> Path:
    """Convert a research ``.pth`` checkpoint into a `save_pretrained` directory."""
    model = load_legacy_checkpoint(src, **kwargs)
    return save_pretrained(model, dst, dtype=dtype, source=str(Path(src).name))


def _download(name: str) -> Path:
    entry = PRETRAINED[name]
    target = cache_dir() / name
    if (target / WEIGHTS_NAME).exists():
        return target
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "Downloading the paper checkpoints needs gdown: pip install 'point-affinity-transformer[pretrained]'"
        ) from e

    target.mkdir(parents=True, exist_ok=True)
    legacy = target / "legacy.pth"
    print(f"Downloading {name} checkpoint (~4.5 GB, includes optimizer state) to {target} ...")
    gdown.download(id=entry["gdrive_id"], output=str(legacy), quiet=False)
    convert_legacy_checkpoint(legacy, target)
    legacy.unlink()
    return target


def load_pretrained(name_or_path: str | os.PathLike, device="cpu") -> PointAffinityTransformer:
    """Load a saved model in eval mode.

    Args:
        name_or_path: A model directory, as written by `save_pretrained`,
            ``pat export`` or a training run (``{output_dir}/model``).
        device: Device to move the model to.
    """
    path = Path(name_or_path)
    if path.is_dir():
        return _load_dir(path, device)
    if path.is_file():
        if path.suffix != ".pth":
            warnings.warn(f"Treating {path} as a legacy research checkpoint", stacklevel=2)
        return load_legacy_checkpoint(path).to(device)
    if str(name_or_path) in PRETRAINED:
        return _load_dir(_download(str(name_or_path)), device)
    raise FileNotFoundError(f"No model directory at {name_or_path!r}")
