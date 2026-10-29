"""Point Affinity Transformer: global neuron shape reasoning on point clouds."""

from importlib.metadata import PackageNotFoundError, version

from . import cluster, metrics
from .inference import EncodedCloud, Segmentation, encode, segment
from .model import PATConfig, PointAffinityTransformer, expand_features
from .presets import PRESETS, create_model
from .pretrained import load_pretrained, save_pretrained

try:
    __version__ = version("point-affinity-transformer")  # defined in pyproject.toml
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"

__all__ = [
    "PATConfig",
    "PointAffinityTransformer",
    "EncodedCloud",
    "Segmentation",
    "encode",
    "segment",
    "create_model",
    "PRESETS",
    "load_pretrained",
    "save_pretrained",
    "expand_features",
    "cluster",
    "metrics",
]
