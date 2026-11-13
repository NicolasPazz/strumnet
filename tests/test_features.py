import numpy as np
import pytest

from strumnet.features import FEATURES, extract_features, feature_vector
from strumnet.synth import FS, WINDOW_S

t = np.arange(int(FS * WINDOW_S)) / FS


def tone(f0, amps):
    return sum(a * np.sin(2 * np.pi * (n + 1) * f0 * t) for n, a in enumerate(amps))


def test_all_features_present_and_finite():
    f = extract_features(tone(110.0, [1.0, 0.5, 0.25]))
    assert tuple(f) == FEATURES
    assert np.isfinite(feature_vector(tone(110.0, [1.0, 0.5]))).all()


@pytest.mark.parametrize("f0", [82.4069, 110.0, 146.83, 196.0, 246.94, 329.63])
def test_peak_frequency_is_accurate(f0):
    f = extract_features(tone(f0, [1.0, 0.4, 0.2]))
    assert f["f0_hz"] == pytest.approx(f0, rel=1e-3)
    assert abs(f["cents_dev"]) < 2


def test_detuned_string():
    f0 = 110.0 * 2 ** (-25 / 1200)
    f = extract_features(tone(f0, [1.0, 0.3]))
    assert f["cents_dev"] == pytest.approx(-25, abs=1.5)
    assert f["cents_note"] == pytest.approx(-25, abs=1.5)


def test_weak_fundamental_shows_up_in_subharmonic_ratio():
    # the 2nd harmonic is louder than the fundamental: the peak is an octave up,
    # but sub2_rel_amp and the second peak give it away
    f = extract_features(tone(110.0, [0.5, 1.0, 0.3]))
    assert f["f0_hz"] == pytest.approx(220.0, rel=1e-3)
    assert f["sub2_rel_amp"] > 0.2
    assert f["peak2_hz"] == pytest.approx(110.0, rel=0.01)
    assert abs(f["cents_note"]) < 2  # still "A", just the wrong octave


def test_clean_tone_has_no_subharmonics():
    f = extract_features(tone(196.0, [1.0, 0.5, 0.3]))
    assert f["sub2_rel_amp"] < 0.05
    assert f["sub3_rel_amp"] < 0.05
