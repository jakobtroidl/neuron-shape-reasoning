"""Point clouds with extra per-point feature columns after xyz."""

import pytest
import torch

import pat
from pat.preprocess import Normalizer, pad_batch


@pytest.fixture
def feature_model():
    torch.manual_seed(0)
    return pat.PointAffinityTransformer(dim=32, num_latents=16, depth=2, heads=2, dim_head=16, feature_dim=2).eval()


def test_model_takes_feature_columns(feature_model):
    assert feature_model.in_dim == 5
    pc = torch.randn(2, 40, 5)
    lat = feature_model.encode(pc)
    assert feature_model.decode(lat, pc[:, :7], pc[:, 7:14]).shape == (2, 7)
    with pytest.raises(ValueError, match="5 input columns"):
        feature_model.encode(torch.randn(2, 40, 3))


def test_features_change_the_output(feature_model):
    pc = torch.randn(1, 40, 5)
    other = pc.clone()
    other[..., 3:] += 1.0
    assert not torch.allclose(feature_model.encode(pc), feature_model.encode(other))


def test_normalizer_and_padding_only_touch_xyz():
    pts = torch.cat([torch.randn(10, 3) * 50 + 100, torch.arange(20.0).view(10, 2)], dim=1)
    norm = Normalizer.fit(pts, scale=10.0)
    out = norm(pts)
    torch.testing.assert_close(out[:, 3:], pts[:, 3:])
    torch.testing.assert_close(out[:, :3].mean(0), torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(norm.inverse(out), pts)

    pc, mask = pad_batch([out, out[:4]])
    assert pc.shape == (2, 10, 5) and mask[1].sum() == 4
    assert (pc[1, 4:, :3] == -1).all() and (pc[1, 4:, 3:] == 0).all()


def test_high_level_api_with_ids(feature_model):
    xyz = torch.randn(200, 3) * 20
    ids = torch.randint(0, 3, (200, 1)).float()
    points = torch.cat([xyz, ids, ids * 2], dim=1)
    enc = pat.encode(feature_model, points, scale=100.0)
    assert enc.affinity().shape == (200, 200)
    assert enc.affinity(points[:5], points[5:9]).shape == (5, 4)
    with pytest.raises(ValueError, match="5 columns"):
        enc.affinity(xyz[:5])
    seg = pat.segment(feature_model, points, threshold=0.9, min_cluster_size=1, max_anchors=50)
    assert seg.labels.shape == (200,) and seg.queries.shape == (200, 5)


def test_expand_features_preserves_outputs(tiny_model):
    expanded = pat.expand_features(tiny_model, 3)
    assert expanded.config.feature_dim == 3 and expanded.in_dim == 6
    pc = torch.randn(1, 30, 3) * 0.1
    q = torch.randn(1, 8, 3) * 0.1
    feats = lambda x: torch.cat([x, torch.randn(*x.shape[:-1], 3)], dim=-1)  # noqa: E731
    with torch.no_grad():
        ref = tiny_model(pc, None, q, q.flip(1))
        out = expanded(feats(pc), None, feats(q), feats(q.flip(1)))
    torch.testing.assert_close(out, ref)


def test_save_load_keeps_feature_dim(feature_model, tmp_path):
    pat.save_pretrained(feature_model, tmp_path / "m")
    assert pat.load_pretrained(tmp_path / "m").config.feature_dim == 2


def test_dataset_with_features(swc_dataset, tmp_path):
    pytest.importorskip("navis")
    from pat.train import MergedNeuronDataset, SkeletonStore
    from pat.train.config import DataConfig
    from pat.train.data import read_root_ids

    swc, csv = swc_dataset

    def neuron_id(root_id, neuron):
        return torch.full((len(neuron.nodes), 1), float(root_id))

    store = SkeletonStore(swc, features=["radius"], feature_fn=neuron_id, cache_dir=tmp_path / "cache")
    nodes, dust = store.load(100)
    assert nodes.shape[1] == 5 and (nodes[:, 3] == 1.0).all() and (nodes[:, 4] == 100).all()
    assert store.load(100)[0].shape == nodes.shape  # from cache

    cfg = DataConfig(swc_dir=str(swc), neuron_ids=str(csv), scale=100.0, samples_per_neuron=16,
                     max_neurons_merged=2, n_dust_neurons=1, n_dust_nodes_per_neuron=4, num_pairs=64,
                     query_mode={"input": 1, "skeleton": 1, "jitter": 1})
    item = MergedNeuronDataset(store, read_root_ids(csv), cfg)[0]
    pc, mask = item["pc"], item["mask"]
    assert pc.shape == (36, 5) and item["qa"].shape == (64, 5)
    assert (pc[~mask, 3:] == 0).all()                       # padding features are zero
    on_neuron = mask & (item["labels"] != item["dust_label"])
    assert set(pc[on_neuron, 4].tolist()) <= set(range(100, 106))  # features are not augmented
    assert set(item["qa"][:, 4].tolist()) <= set(range(100, 106)) | {0.0}  # 0 = padding or synthetic dust


def test_train_with_features_from_3d_init(swc_dataset, tmp_path, tiny_model):
    pytest.importorskip("navis")
    from pat.train import load_config, train

    swc, csv = swc_dataset
    pat.save_pretrained(tiny_model, tmp_path / "init")
    cfg = load_config(None, [
        f"init={tmp_path / 'init'}", "device=cpu", "num_workers=0", f"output_dir={tmp_path / 'out'}",
        "optim.epochs=1", "optim.batch_size=2", "optim.warmup_epochs=0",
        f"data.swc_dir={swc}", f"data.neuron_ids={csv}", "data.features=[radius]", "data.scale=100.0",
        "data.samples_per_neuron=16", "data.max_neurons_merged=2", "data.n_dust_neurons=1",
        "data.n_dust_nodes_per_neuron=4",
    ])
    model = train(cfg)
    assert model.config.feature_dim == 1
    assert pat.load_pretrained(tmp_path / "out" / "model").config.feature_dim == 1
