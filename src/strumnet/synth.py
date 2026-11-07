"""Synthetic plucked-string notes as a phone tuner app would capture them.

What gets simulated, roughly in signal order:

* Plucked string with up to 10 partials. Timbre depends on where the string
  is plucked (comb filter ``|sin(pi * n * beta)|``) and on a random spectral
  tilt, which covers acoustic / electric / nylon-ish sounds.
* Inharmonicity of real strings: ``f_n = n * f0 * sqrt(1 + B * n^2)``.
* Exponential decay, faster for the upper partials, and a short analysis
  window (0.5 s) taken at a random point of the decay.
* Sympathetic resonance: another open string ringing quietly.
* Phone microphone: 2nd-order high-pass with a random cutoff. On the low
  strings this often leaves the 2nd harmonic louder than the fundamental,
  which is exactly what fools a max-peak tuner into an octave error.
* Room noise: white noise at one of four levels, plus sometimes mains hum
  (50 Hz family) and low-frequency fan rumble.
"""

import numpy as np

from .tuning import NOMINAL_HZ, STRINGS

FS = 8000  # sample rate (Hz)
WINDOW_S = 0.5  # analysis window (s)
N_PARTIALS = 10

SYMPATHETIC_P = 0.35
HUM_P = 0.30


def mic_response(f, fc):
    """Magnitude of a 2nd-order high-pass: small phone mics roll off the lows."""
    r = (f / fc) ** 2
    return r / np.sqrt(1 + r**2)


def plucked_note(f0, t, rng, gain=1.0):
    """Sum of decaying inharmonic partials.

    Returns the signal plus the initial amplitude and frequency of each partial
    (used later to tell whether the fundamental ended up weaker than a harmonic).
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


def synthesize(string_idx, cents, noise_level, rng):
    """Render one analysis window for ``string_idx`` detuned by ``cents``.

    Returns ``(signal, weak_fundamental, sympathetic, hum)`` where the three
    flags describe which difficulties ended up in the example.
    """
    t0 = rng.uniform(0.05, 1.0)  # window starts somewhere in the decay
    t = t0 + np.arange(int(FS * WINDOW_S)) / FS
    f0 = NOMINAL_HZ[string_idx] * 2 ** (cents / 1200)

    y, amps, freqs = plucked_note(f0, t, rng)

    sympathetic = rng.random() < SYMPATHETIC_P
    if sympathetic:
        other = rng.choice([i for i in range(len(STRINGS)) if i != string_idx])
        f_other = NOMINAL_HZ[other] * 2 ** (rng.uniform(-15, 15) / 1200)
        gain = rng.uniform(0.1, 0.5)
        y += plucked_note(f_other, t, rng, gain=gain)[0]

    # Phone mic, applied in the frequency domain
    fc = rng.uniform(70, 220)
    Y = np.fft.rfft(y)
    fr = np.fft.rfftfreq(len(y), 1 / FS)
    Y *= mic_response(np.maximum(fr, 1e-3), fc)
    y = np.fft.irfft(Y, n=len(y))

    amps_mic = amps * mic_response(freqs, fc)
    weak_fundamental = bool(amps_mic[0] < amps_mic[1:].max())

    # Room noise
    rms = np.sqrt(np.mean(y**2))
    y = y + rng.normal(0, noise_level * rms * 4, len(t))
    hum = rng.random() < HUM_P
    if hum:
        f_hum = rng.choice([50.0, 100.0, 150.0]) * rng.uniform(0.98, 1.02)
        y += rms * rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * f_hum * t)
        rumble = np.cumsum(rng.normal(0, 1, len(t)))  # brown noise
        rumble -= np.convolve(rumble, np.ones(200) / 200, mode="same")
        y += rms * rng.uniform(0.1, 0.5) * rumble / (np.std(rumble) + 1e-12)

    return y, weak_fundamental, sympathetic, hum
