from tqdm import tqdm
from src.datasets import build_affinity_dataset, build_reproducible_dataset
import src.misc as misc

import src.models as models
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import io
import math

import src.cluster as clustering
import src.metrics as metrics
import os
import time


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="ae_d1024_m512",
    type=str,
    metavar="MODEL",
    help="Name of model to train",
)
parser.add_argument("--pth", required=True, type=str)
parser.add_argument(
    "--device", default="cuda", help="device to use for training / testing"
)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument("--data_global_scale_factor", type=float, default=1.0)
parser.add_argument("--types_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True, help="logs dir path")
parser.add_argument("--batch_size", default=160, type=int, help="batch_size")
parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
parser.add_argument(
    "--point_cloud_size", default=2048, type=int, help="point_cloud_size"
)
parser.add_argument("--depth", default=24, type=int, help="model depth")
parser.add_argument(
    "--thresholds",
    type=str,
    help='Array passed as a comma-separated string, e.g. "1,2,3,4"',
    default="0.5,0.9",
)
parser.add_argument(
    "--store_tensors", action="store_true", help="store tensors for visualization"
)
parser.add_argument(
    "--qual_results",
    action="store_true",
    help="uses reproducable dataset for rendering of qualitative results",
)
parser.add_argument("--fam_to_id_mapping", type=str, required=True)
parser.add_argument("--translate_augmentation", type=float, default=20.0)
parser.add_argument(
    "--max_neurons_merged",
    type=int,
    default=4,
    help="max neurons merged into a single point cloud",
)

args = parser.parse_args()


def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = models.__dict__[args.model](N=args.point_cloud_size, depth=args.depth)
    device = torch.device(args.device)
    model.eval()
    module = torch.load(args.pth, map_location="cpu")["model"]
    model.load_state_dict(module, strict=True)
    model.to(device)

    print(model)

    dataset_test = build_affinity_dataset(
        neuron_path=args.data_path,
        root_id_path=args.types_path,
        samples_per_neuron=args.point_cloud_size,
        scale=args.data_global_scale_factor,
        fam_to_id=args.fam_to_id_mapping,
        translate=args.translate_augmentation,
        max_neurons_merged=args.max_neurons_merged,
        train=False,
    )

    print("max neurons merged", args.max_neurons_merged)

    if args.qual_results:
        reproducible_path = os.path.join(
            os.path.dirname(args.types_path), "reproducable_items"
        )
        dataset_test = build_reproducible_dataset(path=reproducible_path)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.thresholds.count(",") > 0:
        thresholds = [float(i) for i in args.thresholds.split(",")]
    else:
        thresholds = [float(args.thresholds)]

    affinity_criterion = torch.nn.BCELoss()
    affinity_losses = []
    all_metrics = []

    n_chunks = 50  # split the pairs tensor into n_chunks to avoid OOM errors
    best_voi = 1e6
    best_threshold = -1

    string_buffer = io.StringIO()

    start = time.time()

    for t in thresholds:
        with torch.no_grad():
            torch.cuda.empty_cache()
            for i, (
                pc,
                labels,
                mask,
                root_ids,
                queries,
                pairs,
                pairs_labels,
            ) in enumerate(tqdm(data_loader_test)):
                pc = pc.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pairs = pairs.to(device, non_blocking=True)
                pairs_labels = pairs_labels.to(device, non_blocking=True)
                masked_pc = pc[mask]
                masked_labels = labels[mask]

                pairs_chunks = torch.chunk(pairs, n_chunks, dim=1)
                all_affinities = []

                for i, chunk in enumerate(pairs_chunks):
                    # inference the model
                    chunk_pred = model(pc, mask, chunk)
                    pred_affinity = chunk_pred["affinity"]
                    all_affinities.append(pred_affinity)

                pred_affinity = torch.cat(all_affinities, dim=1)
                loss = affinity_criterion(pred_affinity, pairs_labels)

                # reshape dists to 2D
                size = int(math.sqrt(pred_affinity.shape[-1]))
                pred_affinity = pred_affinity.reshape(-1, size, size)

                # mask pred affinity
                pred_aff_masked = pred_affinity[
                    ..., : masked_pc.shape[0], : masked_pc.shape[0]
                ]
                pred_dists = (1.0 - pred_aff_masked).squeeze(0)
                pred_labels = clustering.agglomerative(pred_dists, distance_threshold=t)

                pred_labels = misc.remove_small_clusters(pred_labels, min_size=30)
                pred_labels = pred_labels.view(-1, 1)

                pred_matrix = (pred_labels == pred_labels.T).float()
                gt_matrix = pairs_labels.reshape(pc.shape[1], pc.shape[1])
                gt_matrix = gt_matrix[: pred_matrix.shape[0], : pred_matrix.shape[1]]

                tp, fp, fn, tn = metrics.confusion_matrix(pred_matrix, gt_matrix)
                ars = metrics.adjusted_rand_score(
                    pred_labels.squeeze().cpu().numpy(),
                    masked_labels.squeeze().cpu().numpy(),
                )

                voi, voi_split, voi_merge = metrics.voi(
                    pred_labels.squeeze().cpu().numpy(),
                    masked_labels.squeeze().cpu().numpy(),
                )
                are, are_prec, are_recall = metrics.adjusted_rand_error(
                    pred_labels.squeeze().cpu().numpy(),
                    masked_labels.squeeze().cpu().numpy(),
                )

                all_metrics.append(
                    [
                        tp,
                        fp,
                        fn,
                        tn,
                        voi,
                        ars,
                        voi_split,
                        voi_merge,
                        are,
                        are_prec,
                        are_recall,
                    ]
                )

                mapped_labels = misc.get_mapped_labels(
                    gt_labels=masked_labels, labels=pred_labels
                )  # make sure color assignment is consistent

                if args.store_tensors:
                    images = torch.tensor(pred_aff_masked, dtype=torch.float32)
                    misc.save_images(
                        args.output_dir, images, root_ids, folder="affinity"
                    )
                    misc.save_points(args.output_dir, pc, root_ids, folder="points")
                    misc.save_points(
                        args.output_dir,
                        mapped_labels.unsqueeze(0),
                        root_ids,
                        folder="points",
                        suffix="_labels",
                    )
                    misc.qual_plot(
                        args.output_dir,
                        pc[mask].squeeze(),
                        mapped_labels.squeeze(),
                        root_ids.squeeze(),
                        suffix="_ours",
                        folder="qualitative",
                    )
                    misc.qual_plot(
                        args.output_dir,
                        pc[mask].squeeze(),
                        masked_labels.squeeze(),
                        root_ids.squeeze(),
                        suffix="_gt",
                        folder="qualitative",
                    )
                    misc.qual_plot(
                        args.output_dir,
                        pc[mask].squeeze(),
                        torch.zeros_like(masked_labels.squeeze()),
                        root_ids.squeeze(),
                        suffix="_input",
                        folder="qualitative",
                    )

                affinity_losses.append(loss.item())

        # compute metrics from all batches
        metrics_tensor = torch.tensor(all_metrics)
        tp = torch.sum(metrics_tensor[:, 0])
        fp = torch.sum(metrics_tensor[:, 1])
        fn = torch.sum(metrics_tensor[:, 2])
        tn = torch.sum(metrics_tensor[:, 3])

        voi_mean = torch.mean(metrics_tensor[:, 4])
        ars_mean = torch.mean(metrics_tensor[:, 5])

        voi_split_mean = torch.mean(metrics_tensor[:, 6])
        voi_merge_mean = torch.mean(metrics_tensor[:, 7])
        are_mean = torch.mean(metrics_tensor[:, 8])
        are_prec_mean = torch.mean(metrics_tensor[:, 9])
        are_recall_mean = torch.mean(metrics_tensor[:, 10])

        if voi_mean < best_voi:
            best_voi = voi_mean
            best_threshold = t

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        string_buffer.write(f"-----------------------------\n")
        string_buffer.write(f"Evaluation Results Threshold {t}:\n")
        string_buffer.write(f"Euclidean Distance Clustering:\n")
        string_buffer.write(f"-----------------------------\n")
        string_buffer.write(f"Accuracy: {torch.round(accuracy, decimals=2)}\n")
        string_buffer.write(f"Precision: {torch.round(precision, decimals=2)}\n")
        string_buffer.write(f"Recall: {torch.round(recall, decimals=2)}\n")
        string_buffer.write(f"F1: {torch.round(f1, decimals=2)}\n")
        string_buffer.write(f"-----------------------------\n")
        string_buffer.write(
            f"Variation of Information: {torch.round(voi_mean, decimals=2)}\n"
        )
        string_buffer.write(
            f"Variation of Information Split: {torch.round(voi_split_mean, decimals=2)}\n"
        )
        string_buffer.write(
            f"Variation of Information Merge: {torch.round(voi_merge_mean, decimals=2)}\n"
        )
        string_buffer.write(f"-----------------------------\n")
        string_buffer.write(
            f"Adjusted Rand Error: {torch.round(are_mean, decimals=2)}\n"
        )
        string_buffer.write(
            f"Adjusted Rand Error Precision: {torch.round(are_prec_mean, decimals=2)}\n"
        )
        string_buffer.write(
            f"Adjusted Rand Error Recall: {torch.round(are_recall_mean, decimals=2)}\n"
        )
        string_buffer.write(f"-----------------------------\n")
        string_buffer.write(
            f"Adjusted Rand Score: {torch.round(ars_mean, decimals=2)}\n"
        )
        string_buffer.write(f"-----------------------------")
        string_buffer.write(f"\n")

    end = time.time()

    string_buffer.write(
        f"Best VOI Score: {torch.round(best_voi, decimals=2)}, t: {best_threshold}\n"
    )
    string_buffer.write(f"Dataset: {args.data_path}\n")
    string_buffer.write(f"Dataset: {args.types_path}\n")
    string_buffer.write(f"Model Checkpoint: {args.pth}\n")
    string_buffer.write(f"Model: {args.model}\n")
    string_buffer.write(f"-----------------------------")
    string_buffer.write(f"Time taken: {end - start} seconds\n")
    string_buffer.write(f"\n")
    summary = string_buffer.getvalue()
    string_buffer.close()

    print(summary)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/results.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
