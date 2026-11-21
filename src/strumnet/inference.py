"""Load a trained model directory and run it on signals or WAV files."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_window
from .baseline import peak_picking_rule
from .features import extract_features
from .tuning import CLASSES, STATES, STRINGS, split_class


@dataclass
class Prediction:
    label: str
    string: str
    state: str
    confidence: float
    cents: float  # pitch-class deviation, octave-agnostic
    peak_hz: float  # strongest spectral peak
    peak_picking: str  # what a classic max-peak tuner would have said

    def __str__(self):
        naive = self.peak_picking.replace("_", " ")
        return (
            f"{self.string} {self.state.replace('_', ' ')} "
            f"({round(self.cents):+d} cents, p={self.confidence:.2f})"
            f" | peak at {self.peak_hz:.1f} Hz, peak picking says {naive}"
        )


class Tuner:
    def __init__(self, model, features, mean, scale):
        self.model = model
        self.features = list(features)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)

    @classmethod
    def load(cls, path):
        from tensorflow import keras

        path = Path(path)
        meta = json.loads((path / "scaler.json").read_text())
        model = keras.models.load_model(path / "model.keras")
        return cls(model, meta["features"], meta["mean"], meta["scale"])

    def _probabilities(self, X):
        """(n, 18) class probabilities for either model flavour."""
        p = self.model.predict((X - self.mean) / self.scale, verbose=0)
        if isinstance(p, (list, tuple)):  # two heads: P(string) * P(state)
            return (p[0][:, :, None] * p[1][:, None, :]).reshape(len(X), -1)
        return p

    def predict_signal(self, y):
        feats = extract_features(y)
        X = np.array([[feats[name] for name in self.features]])
        probs = self._probabilities(X)[0]
        idx = int(np.argmax(probs))
        string_idx, state_idx = split_class(idx)
        return Prediction(
            label=CLASSES[idx],
            string=STRINGS[string_idx],
            state=STATES[state_idx],
            confidence=float(probs[idx]),
            cents=float(feats["cents_note"]),
            peak_hz=float(feats["f0_hz"]),
            peak_picking=CLASSES[int(peak_picking_rule(feats["f0_hz"], feats["cents_dev"]))],
        )

    def predict_wav(self, path):
        return self.predict_signal(load_window(path))
