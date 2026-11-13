"""Spectral features computed the way a tuner app would, from the raw window.

The first six are what a classic tuner already has at hand (strongest peak,
cents off the nearest string, a couple of harmonic ratios). The other six are
there so the model can notice when the strongest peak is *not* the
fundamental: energy at f0/2 and f0/3, the second strongest peak, the
pitch-class deviation (octave-agnostic) and the spectral centroid.
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
    "cents_note",  # cents off the nearest chromatic note, any octave
    "sub2_rel_amp",  # |X(f0/2)| / |X(f0)|
    "sub3_rel_amp",  # |X(f0/3)| / |X(f0)|
    "peak2_hz",  # second strongest peak, away from the first one
    "peak2_rel_amp",  # second peak / first peak
    "centroid_hz",  # spectral centroid inside the search band
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
    semitones = 12 * np.log2(f0 / 440.0)
    cents_note = 100 * (semitones - np.round(semitones))

    # Second peak, outside the first one's neighbourhood
    excl = np.abs(fr - f0) <= max(15.0, 0.05 * f0)
    k2 = np.argmax(np.where(band & ~excl, mag, 0))

    near = np.abs(fr - f0) <= 20
    P = mag[near] ** 2
    bandwidth = np.sqrt(np.sum((fr[near] - f0) ** 2 * P) / np.sum(P))
    Pb = mag[band] ** 2
    centroid = np.sum(fr[band] * Pb) / np.sum(Pb)

    return {
        "f0_hz": f0,
        "cents_dev": cents_dev,
        "h1_rel_amp": _amp_near(mag, fr, 2 * f0) / amp0,
        "h2_rel_amp": _amp_near(mag, fr, 3 * f0) / amp0,
        "energy": float(np.mean(y**2)),
        "bandwidth_hz": bandwidth,
        "cents_note": cents_note,
        "sub2_rel_amp": _amp_near(mag, fr, f0 / 2) / amp0,
        "sub3_rel_amp": _amp_near(mag, fr, f0 / 3) / amp0,
        "peak2_hz": fr[k2],
        "peak2_rel_amp": mag[k2] / amp0,
        "centroid_hz": centroid,
    }


def feature_vector(y, fs=FS):
    f = extract_features(y, fs)
    return np.array([f[name] for name in FEATURES], dtype=float)
