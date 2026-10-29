import pytest
import torch

pytest.importorskip("navis")
pytest.importorskip("yaml")

from pat.train import MergedNeuronDataset, SkeletonStore, evaluate, load_config, train  # noqa: E402
from pat.train.config import DataConfig  # noqa: E402
from pat.train.data import read_root_ids  # noqa: E402


def small_data_cfg(swc, csv, **kw):
    return DataConfig(swc_dir=str(swc), neuron_ids=str(csv), scale=100.0, samples_per_neuron=32,
                      max_neurons_merged=3, n_dust_neurons=2, n_dust_nodes_per_neuron=4, **kw)


@pytest.mark.parametrize("modes", [{"input": 1.0}, {"input_valid": 1, "skeleton": 1, "jitter": 1}])
def test_dataset_items(swc_dataset, tmp_path, modes):
    swc, csv = swc_dataset
    cfg = small_data_cfg(swc, csv, query_mode=modes, num_pairs=200)
    store = SkeletonStore(swc, cache_dir=tmp_path / "cache")
    ds = MergedNeuronDataset(store, read_root_ids(csv), cfg)
    item = ds[0]
    assert item["pc"].shape == (ds.cloud_size, 3) and ds.cloud_size == 32 * 3 + 8
    assert item["qa"].shape == (200, 3) and item["target"].shape == (200,)
    assert item["root_ids"][0] == 100
    n_valid = int(item["mask"].sum())
    assert (item["labels"][:n_valid] > 0).all() and (item["labels"][n_valid:] == 0).all()
    assert len(list((tmp_path / "cache").glob("*.pt"))) > 0


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: ae_d64_m64\ndata:\n  scale: 10.0\n")
    cfg = load_config(p, ["optim.lr=0.5", "data.query_mode={input: 1, skeleton: 1}"])
    assert cfg.model == "ae_d64_m64" and cfg.data.scale == 10.0 and cfg.optim.lr == 0.5
    assert cfg.data.query_mode == {"input": 1, "skeleton": 1}
    with pytest.raises(KeyError):
        load_config(None, ["optim.typo=1"])


def test_train_and_evaluate(swc_dataset, tmp_path):
    swc, csv = swc_dataset
    cfg = load_config(None, [
        "model=ae_d64_m64", "depth=1", "device=cpu", "num_workers=0", f"output_dir={tmp_path / 'out'}",
        "optim.epochs=2", "optim.batch_size=2", "optim.accum_iter=1", "optim.warmup_epochs=0", "optim.lr=1e-3",
    ])
    cfg.data = small_data_cfg(swc, csv, query_mode={"input_valid": 1.0})
    torch.manual_seed(0)
    model = train(cfg)
    assert (tmp_path / "out" / "model" / "model.safetensors").exists()
    assert (tmp_path / "out" / "ckpt" / "checkpoint-1.pt").exists()

    from pat.cli import main
    main(["export", str(tmp_path / "out" / "ckpt" / "checkpoint-1.pt"), str(tmp_path / "exported")])

    ds = MergedNeuronDataset(SkeletonStore(swc), read_root_ids(csv), cfg.data, train=False)
    res = evaluate(model.eval(), ds, thresholds=(0.5, 0.8), min_cluster_size=1, limit=2, progress=False)
    assert set(res) == {0.5, 0.8} and "voi" in res[0.5]
