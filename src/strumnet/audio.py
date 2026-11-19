"""WAV I/O and picking an analysis window out of a real recording."""

import wave
from math import gcd

import numpy as np

from .synth import FS, WINDOW_S


def read_wav(path):
    """Return ``(signal in [-1, 1], sample rate)``. 16-bit mono PCM only for now."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("only 16-bit mono WAV files are supported")
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, "<i2") / 2**15, rate


def write_wav(path, x, rate=FS):
    """16-bit mono. Peak-normalizes only if the signal would clip."""
    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x))
    if peak > 1:
        x = x / peak
    pcm = np.round(x * (2**15 - 1)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def resample(x, rate, target=FS):
    if rate == target:
        return np.asarray(x, dtype=float)
    from scipy.signal import resample_poly

    g = gcd(rate, target)
    return resample_poly(x, target // g, rate // g)


def analysis_window(x, fs=FS, length_s=WINDOW_S, skip_s=0.1, frame_s=0.02):
    """Window of ``length_s`` seconds starting a bit after the attack.

    The attack is found as the loudest short frame. Right at the pluck the
    pitch hasn't settled and there's a lot of broadband noise, so we skip
    ``skip_s`` seconds past it (or as much as the recording allows).
    """
    n = int(fs * length_s)
    if len(x) < n:
        raise ValueError(f"recording is shorter than {length_s} s")
    frame = int(fs * frame_s)
    n_frames = len(x) // frame
    energy = (x[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
    onset = int(np.argmax(energy)) * frame
    start = min(onset + int(skip_s * fs), len(x) - n)
    return np.asarray(x[start : start + n], dtype=float)


def load_window(path):
    """Read a WAV file and return a single analysis window at :data:`FS`."""
    x, rate = read_wav(path)
    return analysis_window(resample(x, rate))
