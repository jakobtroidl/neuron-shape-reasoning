import sys
from pathlib import Path

import pytest
import torch

import pat
from pat.preprocess import pad_batch


def test_shapes(tiny_model):
    pc = torch.randn(2, 50, 3) * 0.1
    lat = tiny_model.encode(pc)
    assert lat.shape == (2, 16, 32)
    q = torch.randn(2, 7, 3) * 0.1
    aff = tiny_model.decode(lat, q, q.flip(1))
    assert aff.shape == (2, 7) and (aff >= 0).all() and (aff <= 1).all()
    logits = tiny_model.decode(lat, q, q.flip(1), logits=True)
    torch.testing.assert_close(logits.sigmoid(), aff)


def test_padding_is_ignored(tiny_model):
    pts = torch.randn(40, 3) * 0.1
    pc, mask = pad_batch([pts], size=64)
    pc2 = pc.clone()
    pc2[0, 40:] = 123.0  # different padding values
    torch.testing.assert_close(tiny_model.encode(pc, mask), tiny_model.encode(pc2, mask))
    torch.testing.assert_close(tiny_model.encode(pc, mask), tiny_model.encode(pts[None]), atol=1e-5, rtol=1e-5)


def test_chunking_does_not_change_affinity(tiny_model):
    pts = torch.randn(60, 3) * 0.3
    enc = pat.encode(tiny_model, pts)
    full = enc.affinity(chunk_size=10**6)
    torch.testing.assert_close(enc.affinity(chunk_size=7), full)
    q, k = torch.randn(5, 3) * 0.3, torch.randn(9, 3) * 0.3
    rect = enc.affinity(q, k, chunk_size=4)
    assert rect.shape == (5, 9)
    pairs = torch.stack([q[:, None].expand(5, 9, 3), k[None].expand(5, 9, 3)], dim=2).reshape(-1, 2, 3)
    torch.testing.assert_close(enc.pair_affinity(pairs, chunk_size=3), rect.reshape(-1))


def test_coordinates_are_used_as_given(tiny_model):
    pts = torch.randn(30, 3) * 0.3 + 0.5
    enc = pat.encode(tiny_model, pts)
    torch.testing.assert_close(enc.points, pts)
    torch.testing.assert_close(enc.latents, tiny_model.encode(pts[None]))
    # Querying the encoded points by coordinate equals the default query set.
    torch.testing.assert_close(enc.affinity(pts), enc.affinity())


def test_subsampling_large_clouds(tiny_model):
    pts = torch.randn(1000, 3)
    enc = pat.encode(tiny_model, pts, max_points=100, generator=torch.Generator().manual_seed(0))
    assert enc.points.shape == (100, 3)
    torch.testing.assert_close(enc.points, pts[enc.index])


LEGACY = Path(__file__).resolve().parents[1] / "research" / "legacy"


def test_matches_legacy_model():
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    sys.path.insert(0, str(LEGACY))
    try:
        import src.models as legacy
    finally:
        sys.path.remove(str(LEGACY))
    torch.manual_seed(0)
    old = legacy.create_pat(dim=64, num_latents=32, depth=2).eval()
    new = pat.PointAffinityTransformer(dim=64, num_latents=32, depth=2).eval()
    new.load_state_dict(old.state_dict(), strict=True)

    pc = torch.randn(2, 100, 3) * 0.2
    mask = torch.rand(2, 100) > 0.3
    pairs = torch.randint(0, 100, (2, 500, 2))
    batch = torch.arange(2)[:, None]
    with torch.no_grad():
        a_old = old(pc, mask, pairs)["affinity"]
        a_new = new(pc, mask, pc[batch, pairs[..., 0]], pc[batch, pairs[..., 1]])
    torch.testing.assert_close(a_new, a_old, atol=1e-5, rtol=1e-4)
