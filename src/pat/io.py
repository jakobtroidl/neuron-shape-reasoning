"""Reading neuron skeletons. Requires the ``swc`` extra (navis)."""

from __future__ import annotations

import numpy as np


def _navis():
    try:
        import navis
    except ImportError as e:
        raise ImportError("Reading SWC files needs navis: pip install 'point-affinity-transformer[swc]'") from e
    return navis


def load_neuron(path, units: str = "um"):
    """Read an SWC file as a ``navis.TreeNeuron``.

    Coordinates are converted to ``units`` if the file declares its units
    (as navis-written files do); otherwise they are used as they are.
    """
    navis = _navis()
    neuron = navis.read_swc(str(path))
    if neuron.units is not None and not neuron.units.dimensionless:
        neuron = neuron.convert_units(units)
    return neuron


def read_swc(path, units: str = "um") -> np.ndarray:
    """Read the node coordinates of an SWC file as an ``(N, 3)`` array (see `load_neuron`)."""
    return load_neuron(path, units).nodes[["x", "y", "z"]].to_numpy(dtype=np.float32)


def read_swcs(paths, units: str = "um") -> tuple[np.ndarray, np.ndarray]:
    """Read several SWC files into one cloud.

    Returns:
        ``(points, labels)`` where ``labels[i]`` is the index of the file that
        point ``i`` came from.
    """
    clouds = [read_swc(p, units) for p in paths]
    labels = np.concatenate([np.full(len(c), i) for i, c in enumerate(clouds)])
    return np.concatenate(clouds), labels
