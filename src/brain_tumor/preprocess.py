"""Image preprocessing shared by training, evaluation AND the web app.

Keeping one implementation here is what guarantees the model sees the same
kind of input at serving time that it saw during training (the original app
resized to 224x224 RGB while the model was trained on 128x128 grayscale).
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_to_brain(gray: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of the largest bright region (the skull/brain).

    The raw scans have black borders of very different widths; cropping makes
    the brain fill the frame consistently. Falls back to the full image if no
    contour is found.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 0.2 * gray.shape[1] or h < 0.2 * gray.shape[0]:
        return gray  # implausibly small: keep the original
    return gray[y : y + h, x : x + w]


def preprocess_array(img: np.ndarray, image_size: int = 128, crop: bool = True) -> np.ndarray:
    """BGR/gray uint8 image -> float32 array of shape (size, size, 1) in [0, 1].

    Scaling is a fixed /255 per image, not dataset-wide min/max, so the same
    function works for a single uploaded image at inference time.
    """
    if img is None:
        raise ValueError("Image could not be decoded")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if crop:
        gray = crop_to_brain(gray)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return (gray.astype("float32") / 255.0)[..., None]


def load_and_preprocess(path: str, image_size: int = 128, crop: bool = True) -> np.ndarray:
    return preprocess_array(cv2.imread(str(path)), image_size, crop)


def preprocess_bytes(data: bytes, image_size: int = 128, crop: bool = True) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return preprocess_array(cv2.imdecode(arr, cv2.IMREAD_COLOR), image_size, crop)
