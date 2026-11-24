"""Stage 1-2: data ingestion, validation and leakage-safe splitting.

Run:  python -m brain_tumor.data
Output: data/processed/manifest.csv with one row per image and its split.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import load_config, resolve
from .preprocess import crop_to_brain

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ingest(cfg: dict) -> pd.DataFrame:
    """List every image with an explicit label from the config's class_dirs map."""
    raw = resolve(cfg["data"]["raw_dir"])
    rows = []
    for folder, label in cfg["data"]["class_dirs"].items():
        d = raw / folder
        if not d.is_dir():
            raise FileNotFoundError(f"Expected class folder {d}. See data/README.md.")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(p))
            rows.append(
                dict(
                    path=str(p.relative_to(resolve("."))),
                    label=int(label),
                    readable=img is not None,
                    height=None if img is None else img.shape[0],
                    width=None if img is None else img.shape[1],
                    md5=hashlib.md5(p.read_bytes()).hexdigest(),
                )
            )
    return pd.DataFrame(rows)


def _thumb(path: str) -> np.ndarray:
    g = cv2.cvtColor(cv2.imread(str(resolve(path))), cv2.COLOR_BGR2GRAY)
    t = cv2.resize(crop_to_brain(g), (32, 32), interpolation=cv2.INTER_AREA).astype("float32").ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


def assign_duplicate_groups(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Union images that are (near-)identical so they always land in the same split."""
    n = len(df)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    thumbs = np.stack([_thumb(p) for p in df["path"]])
    corr = thumbs @ thumbs.T
    for i, j in zip(*np.where(np.triu(corr, 1) >= threshold)):
        parent[find(i)] = find(j)
    # exact byte duplicates too
    for _, idx in df.groupby("md5").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            parent[find(k)] = find(idx[0])
    df = df.copy()
    df["group"] = [find(i) for i in range(n)]
    return df


def split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratified + group-aware train/val/test split, plus CV folds on train+val."""
    seed, s = cfg["seed"], cfg["split"]
    df = df.copy()
    y, g = df["label"].values, df["group"].values

    n_test = max(2, round(1 / s["test_size"]))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(df, y, g))
    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"

    dev = df.iloc[dev_idx]
    n_val = max(2, round(len(dev) / (s["val_size"] * len(df))))
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    _, val_rel = next(sgkf.split(dev, dev["label"], dev["group"]))
    df.loc[dev.index[val_rel], "split"] = "val"

    # CV folds over the development set (train+val); the test set is never used.
    df["cv_fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=s["cv_folds"], shuffle=True, random_state=seed)
    for k, (_, f_idx) in enumerate(sgkf.split(dev, dev["label"], dev["group"])):
        df.loc[dev.index[f_idx], "cv_fold"] = k
    return df


def validate(df: pd.DataFrame) -> dict:
    """Data-quality report. Raises on problems that would silently corrupt training."""
    report = {
        "n_images": int(len(df)),
        "unreadable": df.loc[~df["readable"], "path"].tolist(),
        "class_counts": {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "exact_duplicate_files": int(df["md5"].duplicated().sum()),
        "near_duplicate_groups": int((df.groupby("group").size() > 1).sum()),
        "images_in_duplicate_groups": int(df.groupby("group")["path"].transform("size").gt(1).sum()),
    }
    mixed = df.groupby("group")["label"].nunique()
    report["groups_with_conflicting_labels"] = [
        df.loc[df["group"] == gid, "path"].tolist() for gid in mixed[mixed > 1].index
    ]
    if "split" in df:
        leak = df.groupby("group")["split"].nunique()
        assert (leak == 1).all(), "A duplicate group spans more than one split"
        report["split_counts"] = (
            df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
        )
    return report


def build_manifest(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = ingest(cfg)
    df = df[df["readable"]].reset_index(drop=True)
    df = assign_duplicate_groups(df, cfg["data"]["duplicate_corr_threshold"])
    df = split(df, cfg)
    report = validate(df)

    out = resolve(cfg["data"]["manifest_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rep_dir = resolve(cfg["artifacts"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "groups_with_conflicting_labels"}, indent=2, default=str))
    if report["groups_with_conflicting_labels"]:
        print("WARNING: near-duplicate images with different labels:", report["groups_with_conflicting_labels"])
    return df


def load_split(df: pd.DataFrame, cfg: dict):
    """Return {split_name: (X, y)} with preprocessing applied."""
    from .preprocess import load_and_preprocess

    p = cfg["preprocess"]
    X = np.stack([load_and_preprocess(resolve(f), p["image_size"], p["crop_to_brain"]) for f in df["path"]])
    return X, df["label"].values.astype("float32")


if __name__ == "__main__":
    build_manifest()
