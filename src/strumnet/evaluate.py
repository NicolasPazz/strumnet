"""Metrics, breakdowns and plots used by the benchmark."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import DIFFICULTIES
from .tuning import CLASSES, split_class


def accuracy_parts(y_true, y_pred):
    """Overall, string-only and state-only accuracy."""
    s_true, st_true = split_class(y_true)
    s_pred, st_pred = split_class(y_pred)
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "acc_string": float(np.mean(s_true == s_pred)),
        "acc_state": float(np.mean(st_true == st_pred)),
    }


def per_class_report(y_true, y_pred):
    rep = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    table = pd.DataFrame(rep).T.loc[list(CLASSES), ["precision", "recall", "support"]]
    table[["precision", "recall"]] *= 100
    table["support"] = table["support"].astype(int)
    return table.round(2)


def by_difficulty(test_df, predictions):
    """Accuracy per method, sliced by the difficulty flags of each example.

    ``predictions`` maps method name -> boolean array (True where it got it right).
    An example can have more than one difficulty.
    """
    hits = pd.DataFrame(predictions, index=test_df.index)
    clean = test_df[list(DIFFICULTIES)].sum(axis=1) == 0
    rows = {"all": hits.mean()}
    for d in DIFFICULTIES:
        mask = test_df[d] == 1
        rows[f"{d} (n={mask.sum()})"] = hits[mask].mean()
    rows[f"clean (n={clean.sum()})"] = hits[clean].mean()
    return (pd.DataFrame(rows).T * 100).round(2)


def by_noise(test_df, predictions):
    hits = pd.DataFrame(predictions, index=test_df.index)
    hits["noise_level"] = test_df["noise_level"]
    return (hits.groupby("noise_level").mean() * 100).round(2)


def plot_by_difficulty(table, path):
    """Grouped bars: one group per slice of the test set, one bar per method."""
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / table.shape[1]
    x = np.arange(len(table))
    colors = ["#a3a3a3", "#2b6cb0", "#dd6b20", "#38a169"]
    for i, col in enumerate(table.columns):
        offset = x + i * width - 0.4 + width / 2
        bars = ax.bar(offset, table[col], width, label=col, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [label.split(" (")[0].replace("_", " ") for label in table.index])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=table.shape[1], loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.imshow(cm, cmap="Blues")
    ticks = np.arange(len(CLASSES))
    ax.set_xticks(ticks, CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(ticks, CLASSES, fontsize=8)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm
