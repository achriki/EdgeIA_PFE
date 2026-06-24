import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import io
from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to best-fp16.tflite")
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


def letterbox(img: Image.Image, size: int = 640, color=(114, 114, 114)):
    """
    Resize `img` to fit within size x size while preserving aspect ratio,
    then pad with grey to fill the rest. Matches YOLOv5's internal
    preprocessing exactly (same padding color, same centering logic).

    ROOT CAUSE FIX: a naive square resize distorts aspect ratio, which the
    model was never trained on — this collapsed confidence to near-zero
    (max objectness ~0.04 instead of ~0.92). Confirmed via debug_output.py.

    Returns:
        canvas   : PIL.Image, size x size, letterboxed
        scale    : float, the resize scale factor applied
        pad_x    : int, horizontal padding added (left side)
        pad_y    : int, vertical padding added (top side)
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), color)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y


def preprocess(image_bytes):
    """
    Letterbox + normalize an incoming image for the TFLite model.

    Returns:
        arr      : np.float32 array, shape (1, H, W, 3), normalized [0,1]
        orig_h   : original image height (for coordinate conversion later)
        orig_w   : original image width
        scale    : letterbox scale factor (needed to undo letterboxing)
        pad_x    : horizontal padding applied (needed to undo letterboxing)
        pad_y    : vertical padding applied (needed to undo letterboxing)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    canvas, scale, pad_x, pad_y = letterbox(img, size=INPUT_W)  # assumes square model input

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr, orig_h, orig_w, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold=0.45):
    """
    Greedy Non-Max Suppression.

    boxes  : np.array shape (N, 4) — x1, y1, x2, y2 (any consistent coord space)
    scores : np.array shape (N,)   — confidence per box
    Returns indices (into the original arrays) of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # highest confidence first

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # IoU of box i against all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        # Keep only boxes with low overlap with the box we just kept
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]  # +1 to re-offset into `order`

    return keep


def postprocess(output, orig_h, orig_w, scale, pad_x, pad_y, conf_threshold=0.25, iou_threshold=0.45):
    """
    Parse YOLOv5 TFLite output into detections, converting letterboxed
    coordinates back to original image coordinates, with NMS to collapse
    duplicate overlapping boxes around the same object.

    conf_threshold=0.25 calibrated empirically against best-fp16.tflite via
    threshold sweep (see debug_output.py --sweep): converges to ~13-16
    detections vs. 14 ground-truth persons on the reference test image.
    NOTE: this threshold is tuned for the fp16 model. If switching back to
    best_int8.tflite, INT8 quantization noise means no single threshold
    cleanly matches ground truth (sweep showed 125-191 phantom detections
    even at conf > 0.50) — re-run the sweep against whichever model is
    actually deployed before trusting this default.

    Output columns: [cx, cy, w, h, objectness, class_prob] (single-class model)
    Box coordinates from the model are normalized to the letterboxed
    640x640 canvas — they must be unpadded and unscaled to map back to the
    original image.
    """
    preds = output[0]  # drop batch dim -> (num_boxes, 6)

    # Single-class model: objectness IS the confidence (no separate class
    # gating needed — there's only one class, "person", so don't filter by
    # an int(class_col) check; that column may not be a clean 0/1 index).
    confidences = preds[:, 4]
    above_thresh = np.where(confidences > conf_threshold)[0]

    if len(above_thresh) == 0:
        return []

    # Convert surviving boxes to x1,y1,x2,y2 in ORIGINAL image coordinates
    # before running NMS — IoU must be computed in a consistent coordinate
    # space, and downstream consumers expect original-image coords anyway.
    xyxy_orig = np.zeros((len(above_thresh), 4), dtype=np.float32)
    for out_i, i in enumerate(above_thresh):
        cx, cy, w, h = preds[i, 0], preds[i, 1], preds[i, 2], preds[i, 3]

        cx_px = cx * INPUT_W
        cy_px = cy * INPUT_H
        w_px = w * INPUT_W
        h_px = h * INPUT_H

        x1 = (cx_px - w_px / 2 - pad_x) / scale
        y1 = (cy_px - h_px / 2 - pad_y) / scale
        x2 = (cx_px + w_px / 2 - pad_x) / scale
        y2 = (cy_px + h_px / 2 - pad_y) / scale

        xyxy_orig[out_i] = [
            max(0, min(x1, orig_w)),
            max(0, min(y1, orig_h)),
            max(0, min(x2, orig_w)),
            max(0, min(y2, orig_h)),
        ]

    scores_above = confidences[above_thresh]
    keep_local = nms(xyxy_orig, scores_above, iou_threshold=iou_threshold)

    detections = []
    for local_i in keep_local:
        x1, y1, x2, y2 = xyxy_orig[local_i]
        detections.append({
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
            "conf": round(float(scores_above[local_i]), 3),
            "class": 0,
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
        # FIX: preprocess() now returns 6 values (added scale, pad_x, pad_y
        # needed to undo letterbox padding when mapping detections back to
        # original image coordinates). Previously this unpacked only 3
        # values, silently mismatched against the letterboxed preprocess().
        arr, orig_h, orig_w, scale, pad_x, pad_y = preprocess(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400
    with LATENCY.time():
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    # FIX: pass scale/pad_x/pad_y through so postprocess can correctly
    # convert letterboxed coordinates back to the original image space.
    detections = postprocess(output, orig_h, orig_w, scale, pad_x, pad_y)
    DETECT_COUNT.inc(len(detections))
    return jsonify({"persons": len(detections), "detections": detections}), 200


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    print(f"Starting TFLite inference server on port {args.port}")
    print(f"Model: {MODEL_PATH} ({INPUT_W}x{INPUT_H})")
    app.run(host="0.0.0.0", port=args.port)