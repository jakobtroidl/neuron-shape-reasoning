"""Training configuration, loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """How merged point clouds and query pairs are synthesized.

    Each training sample merges 1 to ``max_neurons_merged`` neurons. Every
    neuron is jittered, centered and shifted by a random offset of length
    ``translate`` µm. Dust (small terminal fragments of other neurons) is added
    as distractors, then the cloud is randomly rotated and divided by ``scale``.
    """

    swc_dir: str = ""
    neuron_ids: str = ""  # CSV with a root_id column; SWCs are {swc_dir}/{root_id}.swc
    eval_neuron_ids: Optional[str] = None
    cache_dir: Optional[str] = None  # cache parsed skeletons as .pt files
    features: list = field(default_factory=list)  # SWC node columns used as point features, e.g. [radius]
    units: str = "um"
    scale: float = 1.0  # coordinates (in `units`) are divided by this; the paper used 659.88367
    samples_per_neuron: int = 1024
    max_neurons_merged: int = 4
    n_dust_neurons: int = 6
    n_dust_nodes_per_neuron: int = 32
    dust_size: float = 8.0
    translate: float = 20.0
    jitter: float = 1.0
    rotate_degrees: float = 200.0
    # Where query points come from, with sampling weights (see pat.train.data):
    # input | input_valid | skeleton | jitter
    query_mode: dict = field(default_factory=lambda: {"input": 1.0})
    num_pairs: Optional[int] = None  # pairs per sample; default: the cloud size
    query_jitter: float = 2.0  # µm, for query_mode "jitter"


@dataclass
class OptimConfig:
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    epochs: int = 1000
    batch_size: int = 64
    accum_iter: int = 2
    weight_decay: float = 0.01  # the paper runs used AdamW's default
    clip_grad: Optional[float] = None


@dataclass
class TrainConfig:
    model: str = "ae_d1024_m512"
    depth: int = 24
    init: Optional[str] = None  # model directory to fine-tune from
    resume: Optional[str] = None  # checkpoint directory ({output_dir}/ckpt/epoch-N) to resume from
    output_dir: str = "./output"
    device: str = "cuda"
    amp: bool = False  # bf16 autocast
    seed: int = 0
    num_workers: int = 4
    save_every: int = 2
    log_every: int = 20
    logger: str = "stdout"  # stdout | wandb
    wandb_project: str = "pat"
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "optim": OptimConfig}


def _section(cls, d: dict) -> dict:
    """Check keys and coerce strings like ``1e-4`` (which YAML reads as str) to numbers."""
    types = {f.name: str(f.type) for f in fields(cls)}
    unknown = set(d) - set(types)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {', '.join(sorted(unknown))}")
    out = dict(d)
    for k, v in d.items():
        if isinstance(v, str) and "float" in types[k]:
            out[k] = float(v)
        elif isinstance(v, str) and "int" in types[k]:
            out[k] = int(v)
        elif isinstance(v, int) and not isinstance(v, bool) and "float" in types[k]:
            out[k] = float(v)
    return out


def config_from_dict(d: dict) -> TrainConfig:
    kwargs = _section(TrainConfig, d)
    for name, cls in _SECTIONS.items():
        if name in kwargs:
            kwargs[name] = cls(**_section(cls, kwargs[name]))
    return TrainConfig(**kwargs)


def load_config(path=None, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config and apply ``key.sub=value`` overrides."""
    import yaml

    d = yaml.safe_load(Path(path).read_text()) if path else {}
    d = d or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = d
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return config_from_dict(d)
