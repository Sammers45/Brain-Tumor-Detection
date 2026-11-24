"""Fast tests for the pieces that were broken in the original project.

Run:  pytest -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import load_config  # noqa: E402
from brain_tumor.preprocess import preprocess_array, preprocess_bytes  # noqa: E402


def _fake_scan(h=300, w=260):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, (180, 180, 180), -1)
    return img


def test_label_map_is_explicit_and_tumour_is_positive():
    cfg = load_config()
    # YAML would turn unquoted yes/no into booleans - make sure keys are strings
    assert cfg["data"]["class_dirs"] == {"no": 0, "yes": 1}
    assert cfg["data"]["class_names"][1] == "tumor"


def test_preprocess_shape_range_and_gray():
    x = preprocess_array(_fake_scan(), 128)
    assert x.shape == (128, 128, 1)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_bytes_and_array_paths_identical():
    img = _fake_scan()
    ok, buf = cv2.imencode(".png", img)
    assert ok
    np.testing.assert_allclose(preprocess_bytes(buf.tobytes(), 128), preprocess_array(img, 128))


def test_undecodable_upload_raises():
    with pytest.raises(ValueError):
        preprocess_bytes(b"not an image", 128)


manifest = ROOT / "data" / "processed" / "manifest.csv"


@pytest.mark.skipif(not manifest.exists(), reason="run `python -m brain_tumor.data` first")
def test_no_duplicate_group_crosses_splits():
    df = pd.read_csv(manifest)
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert (df.groupby("md5")["split"].nunique() == 1).all()
    # both classes present in every split
    assert (df.groupby("split")["label"].nunique() == 2).all()
