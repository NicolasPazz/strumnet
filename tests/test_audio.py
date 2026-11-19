import numpy as np
import pytest

from strumnet.audio import analysis_window, load_window, read_wav, resample, write_wav
from strumnet.features import extract_features
from strumnet.synth import FS, WINDOW_S


def sine(f, seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    return 0.5 * np.sin(2 * np.pi * f * t)


def test_wav_roundtrip(tmp_path):
    x = sine(110.0, 0.6, FS)
    write_wav(tmp_path / "a.wav", x)
    y, rate = read_wav(tmp_path / "a.wav")
    assert rate == FS
    assert np.max(np.abs(x - y)) < 1e-4


def test_resample_keeps_pitch():
    pytest.importorskip("scipy")
    y = resample(sine(146.83, 1.0, 44100), 44100)
    f = extract_features(analysis_window(y))
    assert f["f0_hz"] == pytest.approx(146.83, rel=2e-3)


def test_window_starts_after_the_attack():
    x = np.zeros(FS * 2)
    x[FS // 2 : FS // 2 + 160] = 1.0  # click at 0.5 s
    w = analysis_window(x)
    assert len(w) == int(FS * WINDOW_S)
    assert w.max() == 0.0  # the click itself is skipped


def test_too_short():
    with pytest.raises(ValueError):
        analysis_window(np.zeros(100))


def test_load_window(tmp_path):
    write_wav(tmp_path / "n.wav", sine(82.41, 1.5, FS))
    assert load_window(tmp_path / "n.wav").shape == (int(FS * WINDOW_S),)
