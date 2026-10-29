"""Training loop. Runs on one GPU, or on several when launched with ``torchrun``."""

from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..model import expand_features
from ..presets import create_model
from ..pretrained import load_pretrained, save_pretrained
from .checkpoint import load_checkpoint, save_checkpoint
from .config import OptimConfig, TrainConfig
from .data import MergedNeuronDataset, SkeletonStore, read_root_ids


def lr_at(epoch: float, cfg: OptimConfig) -> float:
    """Linear warmup, then half-cycle cosine decay to ``min_lr``."""
    if epoch < cfg.warmup_epochs:
        return cfg.lr * epoch / cfg.warmup_epochs
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Logger:
    def __init__(self, cfg: TrainConfig, enabled: bool):
        self.enabled = enabled
        self.wandb = None
        if enabled and cfg.logger == "wandb":
            import wandb

            self.wandb = wandb
            wandb.init(project=cfg.wandb_project, config=cfg.to_dict())

    def log(self, values: dict, step: int, echo: bool = False):
        if not self.enabled:
            return
        if self.wandb is not None:
            self.wandb.log(values, step=step)
        if echo:
            print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in values.items()), flush=True)


def _setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return 0, 1, None
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def build_model(cfg: TrainConfig, cloud_size: int, feature_dim: int = 0):
    """Create the model to train, or load ``cfg.init`` and adapt it to ``feature_dim``."""
    overrides = {"scale": cfg.data.scale, "num_points": cloud_size}
    if cfg.init:
        model = load_pretrained(cfg.init)
        if model.config.feature_dim != feature_dim:
            model = expand_features(model, feature_dim)
        model.config.scale, model.config.num_points = cfg.data.scale, cloud_size
        return model
    return create_model(cfg.model, depth=cfg.depth, feature_dim=feature_dim, **overrides)


def train_one_epoch(model, loader, optimizer, epoch, cfg: TrainConfig, device, logger: Logger, step: int):
    model.train()
    opt = cfg.optim
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for it, batch in enumerate(loader):
        update = (it + 1) % opt.accum_iter == 0 or it + 1 == len(loader)
        if it % opt.accum_iter == 0:
            lr = lr_at(epoch + it / len(loader), opt)
            for group in optimizer.param_groups:
                group["lr"] = lr

        pc, mask, qa, qb, target = (batch[k].to(device, non_blocking=True)
                                    for k in ("pc", "mask", "qa", "qb", "target"))
        sync = contextlib.nullcontext() if update or not hasattr(model, "no_sync") else model.no_sync()
        with sync, torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            logits = model(pc, mask, qa, qb, logits=True)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is {loss.item()} at epoch {epoch}, iteration {it}")
            (loss / opt.accum_iter).backward()

        if update:
            if opt.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        logger.log({"loss": loss.item(), "lr": lr}, step, echo=it % cfg.log_every == 0)
    if logger.enabled:
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s", flush=True)
    return step


def train(cfg: TrainConfig, feature_fn=None):
    """Train (or fine-tune) a PAT model as described by ``cfg``.

    Writes training checkpoints to ``{output_dir}/ckpt/`` and the latest model
    in `save_pretrained` format to ``{output_dir}/model/``.

    Args:
        cfg: Training configuration.
        feature_fn: Optional per-node feature function, see
            [`SkeletonStore`][pat.train.SkeletonStore]. Its columns follow ``cfg.data.features``.
    """
    rank, world, device = _setup_distributed()
    device = device or torch.device(cfg.device)
    torch.manual_seed(cfg.seed + rank)
    out = Path(cfg.output_dir)

    store = SkeletonStore(cfg.data.swc_dir, cfg.data.units, cfg.data.dust_size, cfg.data.cache_dir,
                          features=cfg.data.features, feature_fn=feature_fn)
    dataset = MergedNeuronDataset(store, read_root_ids(cfg.data.neuron_ids), cfg.data, train=True)
    feature_dim = dataset[0]["pc"].shape[1] - 3
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    model = build_model(cfg, dataset.cloud_size, feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    start_epoch = 0
    if cfg.resume:
        ckpt = load_checkpoint(cfg.resume)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    raw_model = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])

    logger = Logger(cfg, enabled=rank == 0)
    if rank == 0:
        n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        print(f"{cfg.model}: {n_params:.1f}M parameters, {feature_dim} point features, {len(dataset)} neurons, "
              f"{world} process(es), effective batch {cfg.optim.batch_size * cfg.optim.accum_iter * world}")

    step = start_epoch * len(loader)
    for epoch in range(start_epoch, cfg.optim.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        step = train_one_epoch(model, loader, optimizer, epoch, cfg, device, logger, step)
        last = epoch + 1 == cfg.optim.epochs
        if rank == 0 and (epoch % cfg.save_every == 0 or last):
            save_checkpoint(out / "ckpt" / f"checkpoint-{epoch}.pt", raw_model, optimizer, epoch, cfg.to_dict())
            save_pretrained(raw_model, out / "model", epoch=epoch)

    if world > 1:
        dist.destroy_process_group()
    return raw_model
