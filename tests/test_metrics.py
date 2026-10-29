import numpy as np
import pytest

from pat.metrics import voi


def test_voi_split_and_merge():
    gt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    split = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # every neuron split in two
    vi, s, m = voi(split, gt)
    assert s == pytest.approx(np.log(2)) and m == pytest.approx(0)
    vi, s, m = voi(np.zeros(8), gt)  # everything merged
    assert s == pytest.approx(0) and m == pytest.approx(np.log(2))


def test_voi_matches_pyvoi():
    pyvoi = pytest.importorskip("pyvoi")
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 500), rng.integers(0, 7, 500)
    # The research code called pyvoi.VI(pred, gt).
    np.testing.assert_allclose(voi(a, b), pyvoi.VI(a, b, torch=False))
