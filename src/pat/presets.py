"""Named model sizes, following the ``ae_d{dim}_m{num_latents}`` scheme of the paper."""

from __future__ import annotations

from .model import PATConfig, PointAffinityTransformer

PRESETS: dict[str, dict] = {
    f"ae_d{d}_m{m}": {"dim": d, "num_latents": m}
    for d in (64, 128, 256, 512, 1024, 2048)
    for m in (64, 256, 512, 1024)
}

DEFAULT_PRESET = "ae_d1024_m512"


def preset_config(name: str = DEFAULT_PRESET, **overrides) -> PATConfig:
    """Return the `PATConfig` of a named preset, with optional field overrides."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {', '.join(PRESETS)}")
    return PATConfig.from_dict({**PATConfig().to_dict(), **PRESETS[name], **overrides})


def create_model(name: str = DEFAULT_PRESET, **overrides) -> PointAffinityTransformer:
    """Create a randomly initialized model from a named preset."""
    return PointAffinityTransformer(preset_config(name, **overrides))
