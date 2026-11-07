"""Synthetic plucked-string notes.

A plucked string with up to 10 partials. Timbre depends on where the string
is plucked (comb filter ``|sin(pi * n * beta)|``) and on a random spectral
tilt. Partials are slightly inharmonic, like real strings:
``f_n = n * f0 * sqrt(1 + B * n^2)``, and decay exponentially, the upper
ones faster. The analysis window (0.5 s) starts at a random point of the
decay, the way a real-time tuner would see it.
"""

import numpy as np

from .tuning import NOMINAL_HZ

FS = 8000  # sample rate (Hz)
WINDOW_S = 0.5  # analysis window (s)
N_PARTIALS = 10


def plucked_note(f0, t, rng, gain=1.0):
    """Sum of decaying inharmonic partials.

    Returns the signal plus the initial amplitude and frequency of each partial.
    """
    beta = rng.uniform(0.06, 0.30)  # pluck position, as a fraction of the string length
    tilt = rng.uniform(0.6, 1.8)  # spectral tilt 1/n^tilt
    B = rng.uniform(1e-5, 4e-4)  # inharmonicity coefficient
    tau = rng.uniform(0.4, 2.0)  # decay time constant (s)

    y = np.zeros_like(t)
    amps, freqs = [], []
    for n in range(1, N_PARTIALS + 1):
        fn = n * f0 * np.sqrt(1 + B * n**2)
        if fn >= FS / 2:
            break
        a = abs(np.sin(np.pi * n * beta)) / n**tilt
        env = np.exp(-t * n**0.5 / tau)  # upper partials die out first
        y += gain * a * env * np.sin(2 * np.pi * fn * t + rng.uniform(0, 2 * np.pi))
        amps.append(a * env[0])
        freqs.append(fn)
    return y, np.array(amps), np.array(freqs)


def synthesize(string_idx, cents, rng):
    """Render one analysis window for ``string_idx`` detuned by ``cents``."""
    t0 = rng.uniform(0.05, 1.0)  # window starts somewhere in the decay
    t = t0 + np.arange(int(FS * WINDOW_S)) / FS
    f0 = NOMINAL_HZ[string_idx] * 2 ** (cents / 1200)
    y, _, _ = plucked_note(f0, t, rng)
    return y
