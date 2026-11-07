import numpy as np

from strumnet.synth import FS, WINDOW_S, mic_response, plucked_note, synthesize


def test_window_length():
    y, *_ = synthesize(1, 0.0, 0.05, np.random.default_rng(0))
    assert y.shape == (int(FS * WINDOW_S),)
    assert np.isfinite(y).all()


def test_same_seed_same_signal():
    a = synthesize(3, -12.5, 0.10, np.random.default_rng(7))
    b = synthesize(3, -12.5, 0.10, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_flags_are_bools():
    for seed in range(20):
        _, weak, symp, hum = synthesize(0, 5.0, 0.02, np.random.default_rng(seed))
        assert all(isinstance(flag, bool) for flag in (weak, symp, hum))


def test_mic_attenuates_lows_more_than_highs():
    low, high = mic_response(np.array([80.0, 1000.0]), fc=150.0)
    assert low < 0.3 < 0.9 < high <= 1.0


def test_partials_stay_below_nyquist():
    t = np.arange(400) / FS
    _, amps, freqs = plucked_note(329.6, t, np.random.default_rng(1))
    assert len(amps) == len(freqs) <= 10
    assert (freqs < FS / 2).all()


def test_low_strings_often_lose_their_fundamental():
    rng = np.random.default_rng(123)
    weak = [synthesize(0, 0.0, 0.02, rng)[1] for _ in range(60)]
    assert 0.2 < np.mean(weak) < 0.9
