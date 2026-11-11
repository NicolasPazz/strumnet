import numpy as np

from strumnet.baseline import peak_picking_rule, predict_rows
from strumnet.dataset import build
from strumnet.tuning import CLASS_INDEX, NOMINAL_HZ


def test_rule_on_clean_peaks():
    f0 = NOMINAL_HZ * 2 ** (np.array([-20, 0, 20, -5, 30, 0]) / 1200)
    dev = np.array([-20, 0, 20, -5, 30, 0])
    pred = peak_picking_rule(f0, dev)
    names = ["E2_flat", "A2_in_tune", "D3_sharp", "G3_in_tune", "B3_sharp", "E4_in_tune"]
    assert pred.tolist() == [CLASS_INDEX[n] for n in names]


def test_rule_gets_octave_errors_wrong():
    # E2 whose fundamental got eaten by the mic: the peak is at ~165 Hz
    pred = peak_picking_rule(2 * NOMINAL_HZ[0], 0.0)
    assert pred != CLASS_INDEX["E2_in_tune"]


def test_rule_struggles_mostly_on_weak_fundamentals():
    rows = build(per_group=5, seed=0)
    hits = predict_rows(rows) == np.array([CLASS_INDEX[r["label"]] for r in rows])
    weak = np.array([r["weak_fundamental"] for r in rows]) == 1
    assert hits[~weak].mean() > hits[weak].mean() + 0.3
