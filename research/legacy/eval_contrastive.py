from tqdm import tqdm
from src.datasets import build_affinity_dataset
from src.models import EmbProjector
import src.misc as misc

import src.models as models
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from sklearn.neighbors import KNeighborsClassifier
import glob
import src.metrics as metrics
import io
import os


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ae_d1024_m512", type=str)
parser.add_argument("--encoder_pth", required=True, type=str)
parser.add_argument("--deep_set_pth", required=True, type=str)
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--data_global_scale_factor", type=float, default=1.0)
parser.add_argument("--neuron_id_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--batch_size", default=160, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--point_cloud_size", default=2048, type=int)
parser.add_argument("--depth", default=24, type=int)
parser.add_argument("--norm_emb", action="store_true")
parser.add_argument("--k", default=15, type=int)
parser.add_argument("--train_emb_path", required=True, type=str)
parser.add_argument("--fam_to_id_mapping", type=str, required=True)
parser.add_argument("--translate_augmentation", type=float, default=20.0)

args = parser.parse_args()


def load_train_emb(path):
    emb_files = glob.glob(f"{path}/*emb.npy")
    types_files = glob.glob(f"{path}/*type.npy")
    print(f"Found {len(emb_files)} embeddings files")
    print(f"Found {len(types_files)} label files")

    embs = []
    for emb_file in tqdm(emb_files):
        data = np.load(emb_file)
        data = np.expand_dims(data, axis=0)
        embs.append(data)
    embs = np.concatenate(embs, axis=0)

    types = []
    for type_file in tqdm(types_files):
        data = np.load(type_file)
        data = np.expand_dims(data, axis=0)
        types.append(data)
    types = np.concatenate(types, axis=0)

    return embs, types


def main():
    print(args)
    cudnn.benchmark = True

    encoder_model = models.__dict__[args.model](
        N=args.point_cloud_size, depth=args.depth
    )
    device = torch.device(args.device)
    encoder_model.eval()
    module = torch.load(args.encoder_pth, map_location="cpu")["model"]
    encoder_model.load_state_dict(module, strict=True)
    encoder_model.to(device)
    encoder_model.eval()
    print(encoder_model)

    emb_projector = EmbProjector(emb_dim=1024, hidden_dim=256, output_dim=32)
    module = torch.load(args.deep_set_pth, map_location="cpu")["model"]
    emb_projector.load_state_dict(module, strict=True)
    emb_projector.to(device)
    emb_projector.eval()
    print(emb_projector)

    dataset_test = build_affinity_dataset(
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
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    norm = True if args.norm_emb else False

    test_emb = []
    test_types = []

    for i, (pc, labels, mask, root_ids, types, pairs, pairs_labels) in enumerate(
        tqdm(data_loader_train)
    ):
        pc = pc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        types = types.to(device, non_blocking=True)
        pairs = pairs.to(device, non_blocking=True)

        with torch.no_grad():  # Do not calculate gradients for model_B
            out = encoder_model(pc, mask, pairs)
            emb = out["latents"]
            contr_emb = emb_projector(emb, normalize=norm)

        test_emb.append(contr_emb.cpu().numpy())
        test_types.append(types.cpu().numpy())

        misc.save_points(
            args.output_dir, contr_emb, root_ids, suffix="emb", folder="emb"
        )
        misc.save_points(args.output_dir, types, root_ids, suffix="type", folder="emb")

    # eval the test embeddings using a KNN classifier
    test_emb = np.concatenate(test_emb, axis=0)
    test_types = np.concatenate(test_types, axis=0)

    train_emb, train_labels = load_train_emb(args.train_emb_path)

    classifier = KNeighborsClassifier(n_neighbors=args.k)
    classifier.fit(train_emb, train_labels)
    y_pred = classifier.predict_proba(test_emb)

    test_types = test_types.squeeze()

    mAP = metrics.mAP(test_types, y_pred)
    top_1_error = metrics.top_k_error(test_types, y_pred, k=1)
    top_5_error = metrics.top_k_error(test_types, y_pred, k=5)

    string_buffer = io.StringIO()
    string_buffer.write(f"-----------------------------\n")
    string_buffer.write(f"Evaluation Results Metric Learning Cell Typing:\n")
    string_buffer.write(f"mAP: {mAP}\n")
    string_buffer.write(f"Top 1 Error: {top_1_error}\n")
    string_buffer.write(f"Top 5 Error: {top_5_error}\n")
    string_buffer.write(f"-----------------------------\n")
    string_buffer.write(f"k: {args.k}\n")
    string_buffer.write(f"Train Emb Path: {args.train_emb_path}\n")
    string_buffer.write(f"Test Emb Path: {args.output_dir}\n")
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
