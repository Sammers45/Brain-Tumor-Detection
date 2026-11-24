"""Model selection with grouped, stratified 5-fold cross-validation on the
development set (train+val). The test set is NOT touched here.

Run:  python -m brain_tumor.experiments --models logreg cnn lstm
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import load_config, resolve, set_seed
from .data import build_manifest, load_split
from .evaluate import choose_threshold, metrics
from .train import fit_one

KEYS = ["roc_auc", "balanced_accuracy", "sensitivity_recall", "specificity", "f1"]


def _logreg(cfg, X_tr, y_tr, X_val, y_val, X_te):
    """Non-deep baseline: logistic regression on 32x32 pixels."""
    import cv2
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    small = lambda X: np.stack([cv2.resize(x[..., 0], (32, 32), interpolation=cv2.INTER_AREA).ravel() for x in X])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced"))
    clf.fit(small(X_tr), y_tr)
    thr = choose_threshold(y_val, clf.predict_proba(small(X_val))[:, 1], cfg["evaluate"]["threshold_metric"])
    return clf.predict_proba(small(X_te))[:, 1], thr


def cross_validate(cfg, model_names):
    df = pd.read_csv(resolve(cfg["data"]["manifest_path"]))
    dev = df[df["split"] != "test"].reset_index(drop=True)
    X, y = load_split(dev, cfg)
    k = cfg["split"]["cv_folds"]
    results = {}
    for name in model_names:
        fold_scores = []
        for f in range(k):
            set_seed(cfg["seed"] + f)
            te = dev["cv_fold"].values == f
            va = dev["cv_fold"].values == (f + 1) % k  # inner fold for early stopping + threshold
            tr = ~(te | va)
            if name == "logreg":
                p, thr = _logreg(cfg, X[tr], y[tr], X[va], y[va], X[te])
            else:
                model, _, thr = fit_one(cfg, name, X[tr], y[tr], X[va], y[va])
                p = model.predict(X[te], verbose=0).ravel()
            m = metrics(y[te], p, thr)
            fold_scores.append(m)
            print(f"{name} fold {f}: " + ", ".join(f"{kk}={m[kk]:.3f}" for kk in KEYS), flush=True)
        results[name] = {
            kk: {"mean": float(np.mean([s[kk] for s in fold_scores])), "std": float(np.std([s[kk] for s in fold_scores]))}
            for kk in KEYS
        }
        results[name]["folds"] = fold_scores
    return results


def to_markdown(results) -> str:
    lines = ["| Model | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}" for k in KEYS) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "cnn", "lstm"])
    a = ap.parse_args()
    cfg = load_config()
    if not resolve(cfg["data"]["manifest_path"]).exists():
        build_manifest(cfg)
    res = cross_validate(cfg, a.models)
    rep = resolve(cfg["artifacts"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "cv_results.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (rep / "cv_results.md").write_text(md + "\n")
    print(md)
