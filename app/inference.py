import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import io
from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to best_int8.tflite")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

# Load TFLite model
MODEL_PATH = Path(args.model)
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input shape: [1, H, W, C]
INPUT_H = input_details[0]['shape'][1]
INPUT_W = input_details[0]['shape'][2]

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
DETECT_COUNT = Counter("persons_detected_total",   "Total persons detected")
LATENCY = Histogram("inference_latency_seconds", "Inference latency")

# Flask app
app = Flask(__name__)


def preprocess(image_bytes):
    """Resize and normalize frame for TFLite input."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((INPUT_W, INPUT_H))
    arr = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    return arr, orig_h, orig_w


def postprocess(output, orig_h, orig_w, conf_threshold=0.4):
    """
    Parse YOLOv5 TFLite output into detections.
    Output shape: [1, num_detections, 6] → [x1, y1, x2, y2, conf, class]
    """
    detections = []
    for pred in output[0]:
        conf = float(pred[4])
        cls  = int(pred[5])
        if conf < conf_threshold or cls != 0:
            continue
        detections.append({
            "x1": round(float(pred[0]) * orig_w, 1),
            "y1": round(float(pred[1]) * orig_h, 1),
            "x2": round(float(pred[2]) * orig_w, 1),
            "y2": round(float(pred[3]) * orig_h, 1),
            "conf": round(conf, 3),
            "class": cls,
        })
    return detections



@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JPEG/PNG image via multipart form-data (field: 'image').
    Returns JSON with list of person detections.
    """
    REQUEST_COUNT.inc()
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400
    image_bytes = request.files["image"].read()
    try:
        arr, orig_h, orig_w = preprocess(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400
    with LATENCY.time():
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    detections = postprocess(output, orig_h, orig_w)
    DETECT_COUNT.inc(len(detections))
    return jsonify({"persons": len(detections), "detections": detections}), 200



@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    print(f"Starting TFLite inference server on port {args.port}")
    print(f"Model: {MODEL_PATH} ({INPUT_W}x{INPUT_H})")
    app.run(host="0.0.0.0", port=args.port)