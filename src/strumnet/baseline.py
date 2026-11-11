"""The classic tuner: take the strongest peak and trust it.

String = nominal pitch closest to the peak, state = +-10 cents threshold on
the deviation from that string. No training involved. It is nearly perfect on
clean signals and falls apart as soon as the mic makes a harmonic louder than
the fundamental, which is the whole reason this project exists.
"""

import numpy as np

from .tuning import IN_TUNE_CENTS, class_index, nearest_string, state_from_cents


def peak_picking_rule(f0_hz, cents_dev, tolerance=IN_TUNE_CENTS):
    """Predicted class index for each example (arrays in, array out)."""
    string_idx = nearest_string(np.asarray(f0_hz, dtype=float))
    state_idx = state_from_cents(cents_dev, tolerance)
    return class_index(string_idx, state_idx)


def predict_rows(rows):
    """Same rule on the row dicts returned by :func:`strumnet.dataset.build`."""
    f0 = np.array([r["f0_hz"] for r in rows], dtype=float)
    dev = np.array([r["cents_dev"] for r in rows], dtype=float)
    return np.atleast_1d(peak_picking_rule(f0, dev))
