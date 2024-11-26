# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import torch

import src.misc as misc
import src.lr_sched as lr_sched
import wandb


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (pc, labels, mask, root_ids, families, pairs, pairs_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        pc = pc.to(device, non_blocking=True)  # query coords
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)  # mask for pc
        pairs = pairs.to(device, non_blocking=True)  # pairs of indices for distance matrix
        pairs_labels = pairs_labels.to(device, non_blocking=True)  # distance matrix

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(pc, mask, pairs)
            pred_aff = outputs['affinity']
            aff_bce = criterion(pred_aff, pairs_labels)
            loss = aff_bce

        loss_value = loss.item()

        wandb.log({"ae_affinity_bce_loss": aff_bce})
        wandb.log({"ae_total_loss": loss_value})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss /= accum_iter

        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=update_grad,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_vol=loss.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_memory_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # Convert to GB
        print(f"GPU {i} max memory allocated: {gpu_memory_allocated:.2f} GB")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
