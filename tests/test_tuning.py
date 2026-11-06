import numpy as np
import pytest

from strumnet.tuning import (
    CLASSES,
    NOMINAL_HZ,
    STRINGS,
    cents,
    class_index,
    nearest_string,
    split_class,
    state_from_cents,
)


def test_eighteen_classes_in_string_major_order():
    assert len(CLASSES) == 18
    assert CLASSES[:3] == ("E2_flat", "E2_in_tune", "E2_sharp")
    assert CLASSES[-1] == "E4_sharp"


def test_class_index_roundtrip():
    idx = np.arange(18)
    s, st = split_class(idx)
    assert np.array_equal(class_index(s, st), idx)


def test_cents_of_an_octave_and_a_semitone():
    assert cents(220.0, 110.0) == pytest.approx(1200)
    assert cents(110.0 * 2 ** (1 / 12), 110.0) == pytest.approx(100)


def test_nearest_string_on_nominal_pitches():
    for i, f in enumerate(NOMINAL_HZ):
        assert nearest_string(f) == i
        assert nearest_string(f * 2 ** (30 / 1200)) == i  # 30 cents sharp is still the same string


def test_nearest_string_is_fooled_by_an_octave():
    # 2nd harmonic of A2 sits between G3 and B3 and lands on G3/B3 territory
    assert STRINGS[nearest_string(2 * NOMINAL_HZ[1])] in ("G3", "B3")


@pytest.mark.parametrize(
    "c, expected",
    [(-30, 0), (-10, 1), (0, 1), (10, 1), (10.01, 2), (45, 2)],
)
def test_state_thresholds(c, expected):
    assert state_from_cents(c) == expected


def test_vectorized_helpers():
    f = np.array([82.4, 110.0, 330.0])
    assert nearest_string(f).tolist() == [0, 1, 5]
    assert state_from_cents(np.array([-20.0, 3.0, 20.0])).tolist() == [0, 1, 2]
