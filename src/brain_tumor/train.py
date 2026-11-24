"""Stage 5-7: train the final model, evaluate once on the held-out test set,
and save a versioned artifact (model + metadata) that the app loads.

Run:  python -m brain_tumor.train --model cnn
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import load_config, resolve, set_seed
from .data import build_manifest, load_split
from .evaluate import bootstrap_ci, choose_threshold, metrics, plot_confusion, plot_history, plot_roc


def class_weights(y) -> dict:
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    return {c: len(y) / (2.0 * counts[c]) for c in (0, 1)}


def fit_one(cfg, name, X_tr, y_tr, X_val, y_val, verbose=0):
    import tensorflow as tf

    from .models import build

    t = cfg["train"]
    model = build(name, X_tr.shape[1:], augment=t["augment"], lr=t["learning_rate"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=t["early_stopping_patience"], restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),  # explicit, stratified val set (not validation_split)
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        class_weight=class_weights(y_tr) if t["use_class_weights"] else None,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    p_val = model.predict(X_val, verbose=0).ravel()
    thr = choose_threshold(y_val, p_val, cfg["evaluate"]["threshold_metric"])
    return model, hist.history, thr


def main(model_name: str | None = None, config_path: str | None = None):
    cfg = load_config(config_path)
    name = model_name or cfg["train"]["model"]
    set_seed(cfg["seed"])

    manifest_path = resolve(cfg["data"]["manifest_path"])
    df = pd.read_csv(manifest_path) if manifest_path.exists() else build_manifest(cfg)
    sets = {s: load_split(df[df["split"] == s], cfg) for s in ("train", "val", "test")}

    model, history, thr = fit_one(cfg, name, *sets["train"], *sets["val"], verbose=2)

    X_te, y_te = sets["test"]
    p_te = model.predict(X_te, verbose=0).ravel()
    test = metrics(y_te, p_te, thr)
    test["ci95"] = bootstrap_ci(y_te, p_te, thr)
    val = metrics(sets["val"][1], model.predict(sets["val"][0], verbose=0).ravel(), thr)

    model_dir, rep = resolve(cfg["artifacts"]["model_dir"]), resolve(cfg["artifacts"]["reports_dir"])
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"brain_tumor_{name}.keras"
    model.save(model_path)

    meta = {
        "model_name": name,
        "model_file": model_path.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_map": {"0": cfg["data"]["class_names"][0], "1": cfg["data"]["class_names"][1]},
        "positive_class": cfg["data"]["class_names"][1],
        "threshold": thr,
        "preprocess": cfg["preprocess"],
        "train_size": int(len(sets["train"][1])),
        "val_metrics": val,
        "test_metrics": test,
        "epochs_trained": len(history["loss"]),
    }
    (model_dir / f"brain_tumor_{name}.json").write_text(json.dumps(meta, indent=2))
    plot_history(history, rep / "figures" / f"{name}_training_curves.png")
    plot_confusion(test["confusion_matrix"], rep / "figures" / f"{name}_confusion_matrix.png")
    plot_roc(y_te, p_te, rep / "figures" / f"{name}_roc.png", label=name)
    print(json.dumps({"threshold": thr, "test": test}, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm", "transfer"], default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    main(a.model, a.config)
