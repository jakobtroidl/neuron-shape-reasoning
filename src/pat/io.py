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


def node_array(neuron, features=()) -> np.ndarray:
    """Node table of a ``navis.TreeNeuron`` as ``(N, 3 + F)``: xyz, then the ``features`` columns."""
    return neuron.nodes[["x", "y", "z", *features]].to_numpy(dtype=np.float32)


def read_swc(path, units: str = "um", features=()) -> np.ndarray:
    """Read the nodes of an SWC file (see `load_neuron`).

    Args:
        path: SWC file.
        units: Coordinate units to convert to.
        features: Node-table columns appended after xyz, e.g. ``["radius"]``.

    Returns:
        Array of shape ``(N, 3 + len(features))``.
    """
    return node_array(load_neuron(path, units), features)


def read_swcs(paths, units: str = "um", features=()) -> tuple[np.ndarray, np.ndarray]:
    """Read several SWC files into one cloud.

    Returns:
        ``(points, labels)`` where ``labels[i]`` is the index of the file that
        point ``i`` came from.
    """
    clouds = [read_swc(p, units, features) for p in paths]
    labels = np.concatenate([np.full(len(c), i) for i, c in enumerate(clouds)])
    return np.concatenate(clouds), labels
