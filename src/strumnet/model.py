"""Small Keras MLPs: one 18-way softmax, or two heads (string x state).

Needs the ``train`` extra (TensorFlow).
"""

import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .tuning import STATES, STRINGS, class_index, split_class

N_CLASSES = len(STRINGS) * len(STATES)


def build_mlp(n_features, hidden=(16, 8), dropout=0.0, lr=1e-3, two_heads=False):
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in hidden:
        x = layers.Dense(units, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    if two_heads:
        string = layers.Dense(len(STRINGS), activation="softmax", name="string")(x)
        state = layers.Dense(len(STATES), activation="softmax", name="state")(x)
        model = keras.Model(inputs, [string, state])
        xent = "sparse_categorical_crossentropy"
        loss = {"string": xent, "state": xent}
    else:
        out = layers.Dense(N_CLASSES, activation="softmax", name="label")(x)
        model = keras.Model(inputs, out)
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss)
    return model


def targets(y, two_heads):
    if not two_heads:
        return y
    string_idx, state_idx = split_class(y)
    return {"string": string_idx, "state": state_idx}


def fit(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden=(16, 8),
    dropout=0.0,
    lr=1e-3,
    two_heads=False,
    epochs=300,
    batch_size=32,
    patience=25,
    seed=42,
):
    """Train with Adam and early stopping on val loss (best weights restored)."""
    keras.utils.set_random_seed(seed)
    model = build_mlp(X_train.shape[1], hidden, dropout, lr, two_heads)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        targets(y_train, two_heads),
        validation_data=(X_val, targets(y_val, two_heads)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stop],
        verbose=0,
    )
    return model, history


def predict_classes(model, X):
    """Class indices (0..17) for either model flavour."""
    p = model.predict(X, verbose=0)
    if isinstance(p, (list, tuple)):
        return class_index(np.argmax(p[0], axis=1), np.argmax(p[1], axis=1))
    return np.argmax(p, axis=1)


def save(model, scaler, features, path):
    """Write ``model.keras`` plus the scaler stats needed at inference time."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "model.keras")
    meta = {
        "features": list(features),
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
    }
    (path / "scaler.json").write_text(json.dumps(meta, indent=2) + "\n")
