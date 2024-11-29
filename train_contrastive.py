from src.datasets import build_affinity_dataset
from src.models import EmbProjector
import src.misc as misc

import src.models as models
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch.optim import AdamW
import wandb

from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import CosineAnnealingLR
import secrets


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ae_d1024_m512", type=str)
parser.add_argument("--pth", required=True, type=str)
parser.add_argument("--device", default="cuda")
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument("--data_global_scale_factor", type=float, default=1.0)
parser.add_argument("--neuron_id_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True, help="logs dir path")
parser.add_argument("--batch_size", default=160, type=int, help="batch_size")
parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
parser.add_argument("--point_cloud_size", default=2048, type=int)
parser.add_argument("--depth", default=24, type=int, help="model depth")
parser.add_argument("--store_tensors", action="store_true")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--norm_emb", action="store_true", help="normalize embeddings")
parser.add_argument("--fam_to_id_mapping", type=str, required=True)
parser.add_argument("--translate_augmentation", type=float, default=20.0)


args = parser.parse_args()


def main():
    print(args)
    cudnn.benchmark = True

    wandb.init(
        project="implicit-neurons",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )

    encoder_model = models.__dict__[args.model](
        N=args.point_cloud_size, depth=args.depth
    )
    device = torch.device(args.device)
    encoder_model.eval()
    module = torch.load(args.pth, map_location="cpu")["model"]
    encoder_model.load_state_dict(module, strict=True)
    encoder_model.to(device)

    print(encoder_model)

    dataset_train = build_affinity_dataset(
        neuron_path=args.data_path,
        root_id_path=args.neuron_id_path,
        samples_per_neuron=args.point_cloud_size,
        scale=args.data_global_scale_factor,
        max_neurons_merged=1,
        train=True,
        fam_to_id=args.fam_to_id_mapping,
        translate=args.translate_augmentation,
        n_dust_neurons=0,
        n_dust_nodes_per_neuron=0,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # initialize trainable model
    emb_projector = EmbProjector(emb_dim=1024, hidden_dim=256, output_dim=32)
    emb_projector.to(device)
    emb_projector.train()

    normalize = False
    if args.norm_emb:
        normalize = True

    contr_loss = losses.ContrastiveLoss()

    optimizer = AdamW(
        emb_projector.parameters(), lr=0.0001, weight_decay=0.01
    )  # Initial learning rate
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )  # Decays LR every 10 epochs

    for epoch in range(args.epochs):
        for i, (pc, labels, mask, root_ids, types, pairs, pairs_labels) in enumerate(
            data_loader_train
        ):
            pc = pc.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            pairs = pairs.to(device, non_blocking=True)
            # pairs_labels = pairs_labels.to(device, non_blocking=True)

            with torch.no_grad():  # Do not calculate gradients for model_B
                out = encoder_model(pc, mask, pairs)
                latents = out["latents"]
            contr_emb = emb_projector(latents, normalize=normalize)

            loss = contr_loss(contr_emb, types.squeeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"contrastive_loss": loss.item()})

            if i % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

            if epoch % 10 == 0:
                rand_hash = []
                for i in range(len(root_ids)):
                    rand_hash.append(
                        secrets.token_hex(10)
                    )  # Generates a 32-character hex string (128-bit random)

                misc.save_points(
                    args.output_dir,
                    contr_emb,
                    root_ids,
                    suffix="emb",
                    folder="emb_ep_{}".format(epoch),
                    rand_hash=rand_hash,
                )
                misc.save_points(
                    args.output_dir,
                    types,
                    root_ids,
                    suffix="type",
                    folder="emb_ep_{}".format(epoch),
                    rand_hash=rand_hash,
                )
                misc.save_model(
                    args, epoch, emb_projector, emb_projector, optimizer, scheduler
                )

        scheduler.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"Learning rate: {scheduler.get_last_lr()}")

        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_memory_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            print(f"GPU {i} max memory allocated: {gpu_memory_allocated:.2f} GB")

        print("-------------------")


if __name__ == "__main__":
    main()
