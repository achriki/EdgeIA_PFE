import argparse
import sys
from pathlib import Path
 
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest
import pathlib

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to best.pt")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()


pathlib.WindowsPath = pathlib.PosixPath

MODEL_PATH = Path(args.model)
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

model = torch.hub.load(
    str(Path(__file__).parent / "yolov5"),   # local yolov5 repo if present
    "custom",
    path=str(MODEL_PATH),
    source="local",
    force_reload=False,
)
model.eval()
model.conf = 0.4   # confidence threshold
model.classes = [0]  # class 0 = person only

# Prometheus metrics
REQUEST_COUNT  = Counter("inference_requests_total", "Total inference requests")
DETECT_COUNT   = Counter("persons_detected_total",   "Total persons detected")
LATENCY        = Histogram("inference_latency_seconds", "Inference latency")

# Flask app
app = Flask(__name__)
 
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200
 
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JPEG/PNG image via multipart form-data (field: 'image').
    Returns JSON with list of detections: [{x1,y1,x2,y2,conf,class}]
    """
    REQUEST_COUNT.inc()
 
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400
 
    file  = request.files["image"]
    buf   = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
 
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400
 
    with LATENCY.time():
        results = model(frame)
 
    detections = []
    for *box, conf, cls in results.xyxy[0].tolist():
        detections.append({
            "x1": round(box[0], 1),
            "y1": round(box[1], 1),
            "x2": round(box[2], 1),
            "y2": round(box[3], 1),
            "conf": round(conf, 3),
            "class": int(cls),
        })
 
    DETECT_COUNT.inc(len(detections))
    return jsonify({"persons": len(detections), "detections": detections}), 200
 
@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; charset=utf-8"}

# Entry point
if __name__ == "__main__":
    print(f"Starting inference server on port {args.port}")
    print(f"Model: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=args.port)
    