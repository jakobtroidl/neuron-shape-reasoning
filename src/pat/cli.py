"""Command line interface: ``pat train | eval | convert``."""

from __future__ import annotations

import argparse


def _train(args):
    from .train import load_config, train

    train(load_config(args.config, args.set))


def _eval(args):
    import json

    import torch

    from .pretrained import load_pretrained
    from .train import MergedNeuronDataset, SkeletonStore, TensorDirDataset, evaluate, format_results, load_config
    from .train.data import read_root_ids

    cfg = load_config(args.config, args.set)
    torch.manual_seed(cfg.seed)
    model = load_pretrained(args.model, device=args.device or cfg.device)
    if args.items:
        dataset = TensorDirDataset(args.items)
    else:
        store = SkeletonStore(cfg.data.swc_dir, cfg.data.units, cfg.data.dust_size, cfg.data.cache_dir,
                              features=cfg.data.features)
        ids = read_root_ids(cfg.data.eval_neuron_ids or cfg.data.neuron_ids)
        dataset = MergedNeuronDataset(store, ids, cfg.data, train=False)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    results = evaluate(model, dataset, thresholds, min_cluster_size=args.min_cluster_size,
                       symmetrize=args.symmetrize, limit=args.limit)
    print(format_results(results))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


def _convert(args):
    import torch

    from .pretrained import load_legacy_checkpoint, save_pretrained

    model = load_legacy_checkpoint(args.checkpoint, preset=args.preset)
    print(save_pretrained(model, args.out, dtype=torch.float16 if args.fp16 else None))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pat", description="Point Affinity Transformer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train or fine-tune a model")
    p.add_argument("config", help="YAML training config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    p.set_defaults(func=_train)

    p = sub.add_parser("eval", help="evaluate segmentation on merged clouds")
    p.add_argument("config", nargs="?", help="YAML config describing the data (not needed with --items)")
    p.add_argument("--model", required=True, help="model directory (from pat train or save_pretrained)")
    p.add_argument("--items", help="directory of pre-generated items (pc_*.pt, labels_*.pt, mask_*.pt)")
    p.add_argument("--thresholds", default="0.8")
    p.add_argument("--min-cluster-size", type=int, default=30)
    p.add_argument("--symmetrize", action="store_true", help="symmetrize affinities before clustering")
    p.add_argument("--limit", type=int, help="evaluate only the first N items")
    p.add_argument("--device")
    p.add_argument("--out", help="write results as JSON")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=_eval)

    p = sub.add_parser("convert", help="convert a research .pth checkpoint into a model directory")
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--preset", help="model preset; overrides the one stored in the checkpoint")
    p.add_argument("--fp16", action="store_true", help="store weights in float16")
    p.set_defaults(func=_convert)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
