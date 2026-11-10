import csv
from collections import Counter

import pytest

from strumnet.dataset import COLUMNS, NOISE_LEVELS, build, save_csv
from strumnet.tuning import CLASSES

PER_GROUP = 5


@pytest.fixture(scope="module")
def rows():
    return build(per_group=PER_GROUP, seed=1)


def test_size_and_balance(rows):
    assert len(rows) == len(CLASSES) * len(NOISE_LEVELS) * PER_GROUP
    assert set(Counter(r["label"] for r in rows).values()) == {len(NOISE_LEVELS) * PER_GROUP}


def test_split_is_stratified_by_class_and_noise(rows):
    groups = Counter((r["label"], r["noise_level"], r["split"]) for r in rows)
    for label in CLASSES:
        for noise in NOISE_LEVELS:
            assert groups[(label, noise, "train")] == 3
            assert groups[(label, noise, "val")] == 1
            assert groups[(label, noise, "test")] == 1


def test_labels_match_true_detune(rows):
    for r in rows:
        assert r["label"] == f"{r['string']}_{r['state']}"


def test_deterministic():
    a = build(per_group=PER_GROUP, seed=3)
    b = build(per_group=PER_GROUP, seed=3)
    assert [r["f0_hz"] for r in a] == [r["f0_hz"] for r in b]


def test_per_group_must_split_evenly():
    with pytest.raises(ValueError):
        build(per_group=7)


def test_csv_roundtrip(rows, tmp_path):
    path = tmp_path / "notes.csv"
    save_csv(rows, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == COLUMNS
        assert sum(1 for _ in reader) == len(rows)
