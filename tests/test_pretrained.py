import argparse

import pytest
import torch

import pat
from pat.cli import main
from pat.pretrained import load_legacy_checkpoint


def test_save_load_roundtrip(tiny_model, tmp_path):
    pat.save_pretrained(tiny_model, tmp_path / "m")
    loaded = pat.load_pretrained(tmp_path / "m")
    assert loaded.config == tiny_model.config
    for (k, a), (_, b) in zip(tiny_model.state_dict().items(), loaded.state_dict().items()):
        torch.testing.assert_close(a, b, msg=k)


def test_load_legacy_checkpoint(tmp_path):
    model = pat.create_model("ae_d64_m64", depth=2)
    args = argparse.Namespace(model="ae_d64_m64", depth=2, data_global_scale_factor=123.0)
    torch.save({"model": model.state_dict(), "optimizer": {}, "epoch": 3, "args": args}, tmp_path / "old.pth")

    legacy = load_legacy_checkpoint(tmp_path / "old.pth")
    assert legacy.config.dim == 64 and legacy.config.depth == 2 and legacy.config.num_latents == 64
    torch.testing.assert_close(legacy.state_dict()["mlp.fc.2.weight"], model.state_dict()["mlp.fc.2.weight"])
    assert load_legacy_checkpoint(tmp_path / "old.pth", preset="ae_d64_m64", depth=2).config.depth == 2

    # pat convert writes a regular model directory.
    main(["convert", str(tmp_path / "old.pth"), str(tmp_path / "new")])
    loaded = pat.load_pretrained(tmp_path / "new")
    torch.testing.assert_close(loaded.state_dict()["mlp.fc.2.weight"], model.state_dict()["mlp.fc.2.weight"])


def test_load_pretrained_rejects_non_directories(tmp_path):
    (tmp_path / "file.pth").write_bytes(b"")
    for path in (tmp_path / "missing", tmp_path / "file.pth"):
        with pytest.raises(FileNotFoundError, match="No model directory"):
            pat.load_pretrained(path)
