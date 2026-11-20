import numpy as np
import pytest

pytest.importorskip("tensorflow")

from strumnet.model import build_mlp, fit, predict_classes  # noqa: E402


def test_parameter_counts():
    assert build_mlp(12, (16, 8)).count_params() == 506
    assert build_mlp(12, (32, 16, 8)).count_params() == 1242
    assert build_mlp(12, (16, 8), two_heads=True).count_params() == 425


@pytest.mark.parametrize("two_heads", [False, True])
def test_fit_and_predict_shapes(two_heads):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 12))
    y = np.arange(90) % 18
    model, history = fit(X, y, X, y, two_heads=two_heads, epochs=3)
    pred = predict_classes(model, X)
    assert pred.shape == (90,)
    assert pred.min() >= 0 and pred.max() < 18
    assert len(history.history["loss"]) == 3


def test_save_load_and_predict(tmp_path):
    from sklearn.preprocessing import StandardScaler

    from strumnet.features import FEATURES, feature_vector
    from strumnet.inference import Tuner
    from strumnet.model import save
    from strumnet.synth import synthesize
    from strumnet.tuning import CLASSES

    rng = np.random.default_rng(0)
    signals = [synthesize(i % 6, 0.0, 0.02, rng)[0] for i in range(36)]
    X = np.array([feature_vector(y) for y in signals])
    y = np.array([(i % 6) * 3 + 1 for i in range(36)])
    scaler = StandardScaler().fit(X)
    model, _ = fit(scaler.transform(X), y, scaler.transform(X), y, two_heads=True, epochs=2)
    save(model, scaler, FEATURES, tmp_path / "m")

    tuner = Tuner.load(tmp_path / "m")
    pred = tuner.predict_signal(signals[0])
    assert pred.label in CLASSES
    assert 0 <= pred.confidence <= 1
