# Training

Training needs the `train` extra and a directory of SWC files plus a CSV with a
`root_id` column (files are expected at `{swc_dir}/{root_id}.swc`).

```bash
pip install "point-affinity-transformer[train]"

pat train research/configs/flywire.yaml                                   # one GPU
torchrun --standalone --nproc_per_node 8 -m pat train research/configs/flywire.yaml   # multi-GPU (DDP)
pat train config.yaml --set optim.lr=5e-5 data.query_mode="{input_valid: 1, skeleton: 1}"
```

A run writes training checkpoints to `{output_dir}/ckpt/checkpoint-{epoch}.pt` and the
latest model to `{output_dir}/model/`, which [`pat.load_pretrained`][pat.load_pretrained]
reads directly. Resume with `--set resume=out/ckpt/checkpoint-100.pt`. Fine-tune from a
paper model with `--set init=flywire`.

## Training samples

Each sample is a synthetic merge error built by
[`MergedNeuronDataset`][pat.train.MergedNeuronDataset]:

1. Pick the anchor neuron plus 0 to `max_neurons_merged - 1` random partners.
2. Jitter, center and shift every neuron by a random offset of length `translate` µm,
   then sample `samples_per_neuron` nodes from each.
3. Add `n_dust_neurons × n_dust_nodes_per_neuron` dust nodes, which are short terminal
   twigs of other neurons, as distractors.
4. Pad to a fixed size, rotate randomly and divide by `scale`.
5. Sample `num_pairs` query pairs. The target is 1 if both lie on the same neuron (not dust).

Query points come from the sources in `query_mode`, mixed by weight:

| mode | queries drawn from |
|---|---|
| `input` | any point of the padded cloud, including padding (paper setting) |
| `input_valid` | non-padding points of the cloud |
| `skeleton` | any node of the merged skeletons, including nodes not in the cloud |
| `jitter` | cloud points plus Gaussian noise of `query_jitter` µm |

The loss is binary cross-entropy on the logits. The optimizer is AdamW with linear warmup
and cosine decay; `amp: true` enables bf16 autocast.

## Configuration

Configs are YAML files mirroring [`TrainConfig`][pat.train.TrainConfig], with `data`
([`DataConfig`][pat.train.DataConfig]) and `optim`
([`OptimConfig`][pat.train.OptimConfig]) sections. The paper settings are in
`research/configs/{flywire,hemibrain,manc}.yaml`. Set `data.cache_dir` to parse every
SWC only once.

## Evaluation

```bash
pat eval research/configs/hemibrain.yaml --model hemibrain --thresholds 0.6,0.7,0.8,0.9
pat eval --model out/model --items data/hemibrain_v1.2/affinity/reproducable_items --out results.json
```

`--items` evaluates the fixed test items shipped with the datasets. Without it, merges are
synthesized from `data.eval_neuron_ids`. Reported metrics are VOI (with split and merge
terms), ARI and pairwise precision, recall and F1. Clustering follows the paper (no
symmetrization) unless `--symmetrize` is given.

## Converting checkpoints

```bash
pat convert old_checkpoint.pth out_dir/        # research-code .pth → model directory
pat export out/ckpt/checkpoint-100.pt out_dir/ # training checkpoint → model directory
```
