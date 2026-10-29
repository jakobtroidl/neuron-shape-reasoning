import argparse

import torch

import pat


def test_save_load_roundtrip(tiny_model, tmp_path):
    pat.save_pretrained(tiny_model, tmp_path / "m")
    loaded = pat.load_pretrained(tmp_path / "m")
    assert loaded.config == tiny_model.config
    for (k, a), (_, b) in zip(tiny_model.state_dict().items(), loaded.state_dict().items()):
        torch.testing.assert_close(a, b, msg=k)


def test_convert_legacy_checkpoint(tmp_path):
    model = pat.create_model("ae_d64_m64", depth=2)
    args = argparse.Namespace(model="ae_d64_m64", depth=2, data_global_scale_factor=123.0)
    torch.save({"model": model.state_dict(), "optimizer": {}, "epoch": 3, "args": args}, tmp_path / "old.pth")

    out = pat.convert_legacy_checkpoint(tmp_path / "old.pth", tmp_path / "new")
    loaded = pat.load_pretrained(out)
    assert loaded.config.dim == 64 and loaded.config.depth == 2 and loaded.config.scale == 123.0
    torch.testing.assert_close(loaded.state_dict()["mlp.fc.2.weight"], model.state_dict()["mlp.fc.2.weight"])
    # Legacy files can also be loaded directly.
    assert pat.load_pretrained(tmp_path / "old.pth").config.scale == 123.0
