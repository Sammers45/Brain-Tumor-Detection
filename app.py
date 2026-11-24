"""Flask app serving the trained brain-tumour classifier (pipeline stage 8).

Run:  python app.py   (after `python -m brain_tumor.train`)

The app reads the model's metadata JSON, so the label mapping, decision
threshold and preprocessing are exactly the ones used in training.
"""
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.preprocess import preprocess_bytes  # noqa: E402

MODEL_NAME = os.environ.get("BRAIN_TUMOR_MODEL", "cnn")
META_PATH = ROOT / "models" / f"brain_tumor_{MODEL_NAME}.json"
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)  # uses ./templates and ./static
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

_model, _meta = None, None


def get_model():
    """Load lazily so the app starts (and /health answers) quickly."""
    global _model, _meta
    if _model is None:
        import tensorflow as tf

        _meta = json.loads(META_PATH.read_text())
        _model = tf.keras.models.load_model(ROOT / "models" / _meta["model_file"])
    return _model, _meta


def predict_bytes(data: bytes) -> dict:
    model, meta = get_model()
    p = meta["preprocess"]
    x = preprocess_bytes(data, p["image_size"], p["crop_to_brain"])[None, ...]
    prob = float(model.predict(x, verbose=0)[0, 0])  # P(tumour)
    positive = prob >= meta["threshold"]
    return {
        "prediction": meta["label_map"]["1"] if positive else meta["label_map"]["0"],
        "tumor_probability": round(prob, 4),
        "threshold": meta["threshold"],
        "message": (
            "Tumour likely - refer for radiologist review"
            if positive
            else "No tumour detected by the model"
        ),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", model=MODEL_NAME, model_available=META_PATH.exists())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify(error="No file uploaded"), 400
    if Path(f.filename).suffix.lower() not in ALLOWED_EXT:
        return jsonify(error="Please upload a .png, .jpg or .jpeg image"), 400
    try:
        # Processed in memory: uploaded scans are not written to disk.
        return jsonify(predict_bytes(f.read()))
    except ValueError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(debug=False)
