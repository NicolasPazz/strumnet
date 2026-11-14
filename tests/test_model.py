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
