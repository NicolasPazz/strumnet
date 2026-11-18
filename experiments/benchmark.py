"""Train the four reference MLPs and compare them with the peak-picking rule.

    python experiments/benchmark.py --data data/guitar_notes.csv

Writes CSV tables to ``results/`` and figures to ``docs/img/``. Everything is
seeded, so re-running on the same machine gives the same numbers (other
TensorFlow builds may differ by a few test examples).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from strumnet import evaluate
from strumnet.baseline import peak_picking_rule
from strumnet.dataset import load_csv
from strumnet.features import FEATURES
from strumnet.model import fit, predict_classes
from strumnet.tuning import CLASS_INDEX

# name, hidden layers, dropout, learning rate, two heads
CONFIGS = [
    ("base", (16, 8), 0.0, 1e-3, False),
    ("wide", (32, 16, 8), 0.0, 1e-3, False),
    ("dropout", (16, 8), 0.2, 5e-3, False),
    ("two_head", (16, 8), 0.0, 1e-3, True),
]


def topology(hidden, two_heads):
    out = "(6+3)" if two_heads else "18"
    return " → ".join([str(len(FEATURES)), *map(str, hidden), out])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/guitar_notes.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="docs/img")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results, figures = Path(args.results), Path(args.figures)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.data)
    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    y_train, y_val, y_test = (d["label"].map(CLASS_INDEX).to_numpy() for d in (train, val, test))

    scaler = StandardScaler().fit(train[list(FEATURES)])
    X_train, X_val, X_test = (scaler.transform(d[list(FEATURES)]) for d in (train, val, test))

    summary, hits = [], {}

    rule = peak_picking_rule(test.f0_hz.to_numpy(), test.cents_dev.to_numpy())
    hits["peak_picking"] = rule == y_test
    evaluate.plot_confusion(
        y_test, rule, "peak-picking rule (test)", figures / "confusion_peak_picking.png"
    )
    summary.append(
        {"model": "peak_picking", "topology": "-", **evaluate.accuracy_parts(y_test, rule)}
    )

    models = {}
    for name, hidden, dropout, lr, two_heads in CONFIGS:
        model, history = fit(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            two_heads=two_heads,
            seed=args.seed,
        )
        models[name] = model
        pred_train = predict_classes(model, X_train)
        pred_test = predict_classes(model, X_test)
        hits[name] = pred_test == y_test

        evaluate.plot_history(history, f"{name}: loss", figures / f"loss_{name}.png")
        evaluate.plot_confusion(
            y_test, pred_test, f"{name} (test)", figures / f"confusion_{name}.png"
        )
        evaluate.per_class_report(y_test, pred_test).to_csv(results / f"per_class_{name}.csv")

        acc_train = float(np.mean(pred_train == y_train))
        parts = evaluate.accuracy_parts(y_test, pred_test)
        summary.append(
            {
                "model": name,
                "topology": topology(hidden, two_heads),
                "params": model.count_params(),
                "epochs": len(history.history["loss"]),
                "acc_train": acc_train,
                **parts,
                "gap": acc_train - parts["acc"],
            }
        )
        print(f"{name:10s} test acc {parts['acc']:.2%}")

    summary = pd.DataFrame(summary).set_index("model")
    summary = summary[
        ["topology", "params", "epochs", "acc_train", "acc", "acc_string", "acc_state", "gap"]
    ]
    summary[["params", "epochs"]] = summary[["params", "epochs"]].astype("Int64")
    pct = ["acc_train", "acc", "acc_string", "acc_state", "gap"]
    summary[pct] = (summary[pct] * 100).round(2)
    summary.to_csv(results / "summary.csv")

    evaluate.by_difficulty(test, hits).to_csv(results / "by_difficulty.csv")
    evaluate.by_noise(test, hits).to_csv(results / "by_noise.csv")

    # Why StandardScaler: relative amplitudes have huge outliers when the
    # fundamental is almost gone, and min-max squashes everything else near 0.
    mm = MinMaxScaler().fit(train[list(FEATURES)])
    Xmm_train, Xmm_val, Xmm_test = (mm.transform(d[list(FEATURES)]) for d in (train, val, test))
    rows = []
    for name, hidden, dropout, lr, two_heads in CONFIGS[:2]:
        model_mm, _ = fit(
            Xmm_train,
            y_train,
            Xmm_val,
            y_val,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            two_heads=two_heads,
            seed=args.seed,
        )
        rows.append(
            {
                "model": name,
                "standard": np.mean(predict_classes(models[name], X_test) == y_test) * 100,
                "minmax": np.mean(predict_classes(model_mm, Xmm_test) == y_test) * 100,
            }
        )
    pd.DataFrame(rows).set_index("model").round(2).to_csv(results / "scaling.csv")

    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()
