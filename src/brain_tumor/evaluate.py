"""Stage 6: evaluation — metrics that matter for a screening task."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true, y_prob, metric: str = "youden") -> float:
    """Pick the decision threshold on VALIDATION data only.

    - "youden": maximise sensitivity + specificity - 1 (default; balanced).
    - "min_sensitivity:0.9": highest threshold whose sensitivity >= 0.9
      (screening setting: cap the miss rate, then maximise specificity).
    - "f1" / "f2": F-beta. Careful: with 61% positives, predicting "tumour"
      for everything already scores F2≈0.89, so F2 tends to collapse
      specificity on this dataset.
    """
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob)
    grid = np.linspace(0.05, 0.95, 91)
    pos, neg = y_true == 1, y_true == 0
    sens = np.array([(y_prob[pos] >= t).mean() for t in grid])
    spec = np.array([(y_prob[neg] < t).mean() for t in grid])
    if metric.startswith("min_sensitivity"):
        target = float(metric.split(":")[1])
        ok = np.where(sens >= target)[0]
        return float(grid[ok[-1]]) if len(ok) else float(grid[0])
    if metric in ("f1", "f2"):
        beta = 1.0 if metric == "f1" else 2.0
        scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    else:
        scores = sens + spec - 1
    # among ties prefer the threshold closest to 0.5 (more stable on small val sets)
    scores = np.asarray(scores)
    best = np.where(np.isclose(scores, scores.max()))[0]
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


def metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, y_prob, threshold, n_boot=2000, seed=0) -> dict:
    """95% bootstrap CIs. With a ~37-image test set these are wide — report them."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob)
    keys = ["roc_auc", "accuracy", "sensitivity_recall", "specificity"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        m = metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}


def plot_confusion(cm: dict, path, title="Confusion matrix (test)"):
    import matplotlib.pyplot as plt

    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap="Blues")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > arr.max() / 2 else "black")
    ax.set_xticks([0, 1], ["no tumor", "tumor"])
    ax.set_yticks([0, 1], ["no tumor", "tumor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, path, label="model"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ["loss", "auc"]):
        ax.plot(history[key], label=f"train {key}")
        ax.plot(history[f"val_{key}"], label=f"val {key}")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
