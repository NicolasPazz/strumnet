"""Spectral features computed the way a tuner app would, from the raw window.

For now these are the numbers a classic tuner already has at hand: the
strongest peak, how many cents it is off the nearest string, a couple of
harmonic ratios, signal energy and how wide the peak is.
"""

import numpy as np

from .synth import FS
from .tuning import NOMINAL_HZ

NFFT = 16384  # zero-padded FFT
F_MIN, F_MAX = 60.0, 1200.0  # search band

FEATURES = (
    "f0_hz",  # strongest spectral peak (parabolic interpolation)
    "cents_dev",  # cents off the string closest to f0_hz
    "h1_rel_amp",  # |X(2*f0)| / |X(f0)|
    "h2_rel_amp",  # |X(3*f0)| / |X(f0)|
    "energy",  # mean signal power
    "bandwidth_hz",  # spread around the peak
)


def _amp_near(mag, fr, f, tol=0.03):
    """Spectrum max within +-tol of ``f`` (tolerates inharmonic partials)."""
    m = (fr >= f * (1 - tol)) & (fr <= f * (1 + tol))
    return mag[m].max() if m.any() else 0.0


def _cents(f, ref):
    return 1200 * np.log2(f / ref)


def extract_features(y, fs=FS):
    """Return a dict with every name in :data:`FEATURES`."""
    w = np.hanning(len(y))
    mag = np.abs(np.fft.rfft(y * w, NFFT))
    fr = np.fft.rfftfreq(NFFT, 1 / fs)
    band = (fr >= F_MIN) & (fr <= F_MAX)
    band_idx = np.where(band)[0]

    # Max peak + parabolic interpolation on the log magnitude
    k = band_idx[np.argmax(mag[band])]
    a, b, c = np.log(mag[k - 1 : k + 2] + 1e-12)
    p = 0.5 * (a - c) / (a - 2 * b + c)
    f0 = (k + p) * fs / NFFT
    amp0 = mag[k]

    i_string = np.argmin(np.abs(np.log2(f0 / NOMINAL_HZ)))
    cents_dev = _cents(f0, NOMINAL_HZ[i_string])

    near = np.abs(fr - f0) <= 20
    P = mag[near] ** 2
    bandwidth = np.sqrt(np.sum((fr[near] - f0) ** 2 * P) / np.sum(P))

    return {
        "f0_hz": f0,
        "cents_dev": cents_dev,
        "h1_rel_amp": _amp_near(mag, fr, 2 * f0) / amp0,
        "h2_rel_amp": _amp_near(mag, fr, 3 * f0) / amp0,
        "energy": float(np.mean(y**2)),
        "bandwidth_hz": bandwidth,
    }


def feature_vector(y, fs=FS):
    f = extract_features(y, fs)
    return np.array([f[name] for name in FEATURES], dtype=float)
