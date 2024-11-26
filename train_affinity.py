from src.datasets import build_affinity_dataset

from src.misc import NativeScalerWithGradNormCount as NativeScaler
from src.engine_affinity import train_one_epoch

import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import wandb
import src.misc as misc
import src.models as models

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def get_args_parser():
    parser = argparse.ArgumentParser("Autoencoder", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument(
        "--accum_iter",
        default=2,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="ae_blob64",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_global_scale_factor", type=float, default=1.0)
    parser.add_argument("--repeat_dataset", type=int, default=1)
    parser.add_argument("--types_path", type=str, required=True)

    parser.add_argument("--point_cloud_size", default=2048, type=int, help="input size")
    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--distributed", action="store_true", help="Run distributed training"
    )

    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output/", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", default=False, type=bool)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--depth", default=24, type=int)
    parser.add_argument("--fam_to_id_mapping", type=str, required=True)
    parser.add_argument("--translate_augmentation", type=float, default=20.0)

    return parser


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # set GPU device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    wandb.init(
        project="implicit-neurons",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
    )

    dataset_train = build_affinity_dataset(
        neuron_path=args.data_path,
        root_id_path=args.types_path,
        samples_per_neuron=args.point_cloud_size,
        scale=args.data_global_scale_factor,
        fam_to_id=args.fam_to_id_mapping,
        translate=args.translate_augmentation,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        shuffle=True,
    )

    model = models.__dict__[args.model](depth=args.depth)

    # If multiple GPUs are available, wrap the model in DataParallel
    if torch.cuda.device_count() > 1 and args.distributed:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    model_without_ddp = model.module if hasattr(model, "module") else model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    print("is distributed: %s" % str(args.distributed))

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = torch.nn.BCELoss()
    print("criterion = %s" % str(criterion))
    print(f"Start training for {args.epochs} epochs")

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch} ...")
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        _ = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            args=args,
        )
        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
