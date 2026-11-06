"""Standard tuning, class labels and pitch helpers."""

import numpy as np

STRINGS = ("E2", "A2", "D3", "G3", "B3", "E4")
NOMINAL_HZ = np.array([82.4069, 110.0, 146.8324, 195.9977, 246.9417, 329.6276])

STATES = ("flat", "in_tune", "sharp")
IN_TUNE_CENTS = 10.0  # |deviation| <= 10 cents counts as in tune

# 18 classes ordered as string * 3 + state, so string = idx // 3 and state = idx % 3
CLASSES = tuple(f"{s}_{st}" for s in STRINGS for st in STATES)
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}


def cents(f, ref):
    """Distance from ``ref`` to ``f`` in cents (100 cents = 1 semitone)."""
    return 1200 * np.log2(np.asarray(f) / ref)


def nearest_string(f_hz):
    """Index of the open string whose nominal pitch is closest (log scale) to ``f_hz``.

    Works on scalars and arrays.
    """
    f = np.asarray(f_hz, dtype=float)
    idx = np.argmin(np.abs(np.log2(f[..., None] / NOMINAL_HZ)), axis=-1)
    return int(idx) if idx.ndim == 0 else idx


def state_from_cents(c, tolerance=IN_TUNE_CENTS):
    """0 = flat, 1 = in tune, 2 = sharp. Works on scalars and arrays."""
    c = np.asarray(c, dtype=float)
    state = np.where(np.abs(c) <= tolerance, 1, np.where(c < 0, 0, 2))
    return int(state) if state.ndim == 0 else state


def class_index(string_idx, state_idx):
    return np.asarray(string_idx) * len(STATES) + np.asarray(state_idx)


def split_class(idx):
    """Inverse of :func:`class_index`: returns ``(string_idx, state_idx)``."""
    idx = np.asarray(idx)
    return idx // len(STATES), idx % len(STATES)
