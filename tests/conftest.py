import pytest
import torch

import pat


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return pat.PointAffinityTransformer(dim=32, num_latents=16, depth=2, heads=2, dim_head=16, num_points=256).eval()


def _write_swc(path, n, offset, seed):
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(n, 3, generator=g) * 2.0
    xyz = steps.cumsum(0) + torch.tensor(offset)
    lines = ["# PointNo Label X Y Z Radius Parent"]
    for i, (x, y, z) in enumerate(xyz.tolist(), start=1):
        parent = -1 if i == 1 else (i - 1 if i % 7 else max(1, i - 5))
        lines.append(f"{i} 0 {x:.3f} {y:.3f} {z:.3f} 1.0 {parent}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def swc_dataset(tmp_path):
    """Six small random-walk skeletons and a root id CSV."""
    swc = tmp_path / "swc"
    swc.mkdir()
    ids = list(range(100, 106))
    for k, rid in enumerate(ids):
        _write_swc(swc / f"{rid}.swc", 80, [10.0 * k, 0, 0], seed=k)
    csv = tmp_path / "ids.csv"
    csv.write_text("root_id\n" + "\n".join(map(str, ids)) + "\n")
    return swc, csv
