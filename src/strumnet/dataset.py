"""Build, save and load the synthetic dataset.

Every (class, noise level) group gets the same number of examples and is
split 60/20/20 into train/val/test, so the split is stratified by both.
Labels come from the *true* pitch; features only see the rendered signal.
"""

import csv

import numpy as np

from .features import FEATURES, extract_features
from .synth import synthesize
from .tuning import IN_TUNE_CENTS, STATES, STRINGS

NOISE_LEVELS = (0.02, 0.05, 0.10, 0.18)
MAX_CENTS = 50.0

# Metadata columns: handy for slicing results, never fed to a model
DIFFICULTIES = ("weak_fundamental", "sympathetic", "hum")
METADATA = ("noise_level", *DIFFICULTIES)
LABELS = ("string", "state", "label", "split")
COLUMNS = FEATURES + METADATA + LABELS


def sample_cents(state, rng):
    if state == "in_tune":
        return rng.uniform(-IN_TUNE_CENTS, IN_TUNE_CENTS)
    c = rng.uniform(IN_TUNE_CENTS, MAX_CENTS)
    return -c if state == "flat" else c


def build(per_group=60, seed=42):
    """Generate the dataset as a list of row dicts.

    ``per_group`` is the number of examples per (class, noise level) and has
    to be a multiple of 5 so the 60/20/20 split is exact.
    """
    if per_group % 5:
        raise ValueError("per_group must be a multiple of 5")

    rng = np.random.default_rng(seed)
    n_test = n_val = per_group // 5
    rows = []
    for i, string in enumerate(STRINGS):
        for state in STATES:
            for noise in NOISE_LEVELS:
                splits = ["test"] * n_test + ["val"] * n_val
                splits += ["train"] * (per_group - len(splits))
                rng.shuffle(splits)
                for split in splits:
                    c = sample_cents(state, rng)
                    y, weak, symp, hum = synthesize(i, c, noise, rng)
                    row = extract_features(y)
                    row.update(
                        noise_level=noise,
                        weak_fundamental=int(weak),
                        sympathetic=int(symp),
                        hum=int(hum),
                        string=string,
                        state=state,
                        label=f"{string}_{state}",
                        split=split,
                    )
                    rows.append(row)
    order = rng.permutation(len(rows))
    return [rows[j] for j in order]


def _fmt(v):
    return f"{float(v):.6g}" if isinstance(v, (float, np.floating)) else v


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(v) for k, v in row.items()})


def load_csv(path):
    """Load the CSV as a pandas DataFrame (pandas comes with the ``train`` extra)."""
    import pandas as pd

    return pd.read_csv(path)
