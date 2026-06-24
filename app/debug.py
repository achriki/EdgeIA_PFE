"""
debug_output.py — Quick diagnostic: inspect raw YOLOv5 TFLite output
before any confidence filtering, to understand the actual column layout.

Run this ON THE PI (same environment as the deployed app) so it uses
tflite_runtime exactly like inference.py does.

Usage:
    python3 debug_output.py --model ./model/best_int8.tflite --image /path/to/test.jpg
"""
import argparse
import numpy as np
from PIL import Image
import io

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--letterbox", action="store_true", help="Use letterbox preprocessing instead of naive square resize")
parser.add_argument("--sweep", action="store_true", help="Sweep confidence thresholds and show detection count after NMS at each one")
args = parser.parse_args()

interpreter = Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_H = input_details[0]['shape'][1]
INPUT_W = input_details[0]['shape'][2]

print(f"Input  shape: {input_details[0]['shape']}  dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}  dtype: {output_details[0]['dtype']}")
print()

with open(args.image, "rb") as f:
    image_bytes = f.read()

img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

if args.letterbox:
    print(">>> Using LETTERBOX preprocessing (aspect-ratio preserved + grey padding)")
    w, h = img.size
    scale = INPUT_W / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (INPUT_W, INPUT_H), (114, 114, 114))
    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    print(f"    scale={scale:.4f}  pad_x={pad_x}  pad_y={pad_y}")
else:
    print(">>> Using NAIVE SQUARE RESIZE preprocessing (old, broken)")
    img_sq = img.resize((INPUT_W, INPUT_H))
    arr = np.array(img_sq, dtype=np.float32) / 255.0

arr = np.expand_dims(arr, axis=0)

interpreter.set_tensor(input_details[0]['index'], arr)
interpreter.invoke()
raw_output = interpreter.get_tensor(output_details[0]['index'])

print(f"Raw output array shape: {raw_output.shape}")
print()

preds = raw_output[0]  # drop batch dim -> (num_boxes, num_cols)
print(f"Number of boxes: {preds.shape[0]}, columns per box: {preds.shape[1]}")
print()

INPUT_W_ = INPUT_W
INPUT_H_ = INPUT_H


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


if args.sweep:
    print("=" * 60)
    print("THRESHOLD SWEEP (detections remaining after NMS at each conf_threshold)")
    print("=" * 60)
    confidences = preds[:, 4]
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        above = np.where(confidences > thresh)[0]
        if len(above) == 0:
            print(f"  conf > {thresh:.2f}  ->  0 raw boxes")
            continue
        xyxy = np.zeros((len(above), 4), dtype=np.float32)
        pad_x_ = pad_x if args.letterbox else 0
        pad_y_ = pad_y if args.letterbox else 0
        scale_ = scale if args.letterbox else 1.0
        for out_i, i in enumerate(above):
            cx, cy, w, h = preds[i, 0], preds[i, 1], preds[i, 2], preds[i, 3]
            cx_px, cy_px = cx * INPUT_W_, cy * INPUT_H_
            w_px, h_px = w * INPUT_W_, h * INPUT_H_
            x1 = (cx_px - w_px / 2 - pad_x_) / scale_
            y1 = (cy_px - h_px / 2 - pad_y_) / scale_
            x2 = (cx_px + w_px / 2 - pad_x_) / scale_
            y2 = (cy_px + h_px / 2 - pad_y_) / scale_
            xyxy[out_i] = [x1, y1, x2, y2]
        scores_above = confidences[above]
        kept = nms(xyxy, scores_above, iou_threshold=0.45)
        print(f"  conf > {thresh:.2f}  ->  {len(above):4d} raw boxes  ->  {len(kept):3d} after NMS")
    print()
    print("Pick the threshold where the post-NMS count stabilizes near your")
    print("known ground-truth count (e.g. 14 for this reference image).")
    import sys
    sys.exit(0)

# Show stats per column across all boxes
print("Column-wise stats (min / max / mean):")
for col in range(preds.shape[1]):
    col_vals = preds[:, col]
    print(f"  col {col}: min={col_vals.min():.4f}  max={col_vals.max():.4f}  mean={col_vals.mean():.4f}")
print()

# Show the single highest-objectness box in full, raw
best_idx = np.argmax(preds[:, 4])
print(f"Highest-objectness box (index {best_idx}):")
print(f"  raw values: {preds[best_idx]}")
print()

# If there's a separate class-prob column (col 5), show combined confidence
if preds.shape[1] >= 6:
    objectness = preds[:, 4]
    class_prob = preds[:, 5]
    combined = objectness * class_prob
    top5 = np.argsort(combined)[-5:][::-1]
    print("Top 5 by objectness * class_prob (RAW, no sigmoid):")
    for i in top5:
        print(f"  idx={i}  objectness={objectness[i]:.4f}  class_prob={class_prob[i]:.4f}  combined={combined[i]:.4f}")

    print()
    print("=" * 60)
    print("SIGMOID-CORRECTED objectness (hypothesis: raw output is pre-sigmoid logits)")
    print("=" * 60)
    obj_sigmoid = 1 / (1 + np.exp(-objectness))
    print(f"  sigmoid(objectness): min={obj_sigmoid.min():.4f}  max={obj_sigmoid.max():.4f}  mean={obj_sigmoid.mean():.4f}")
    n_above_25 = (obj_sigmoid > 0.25).sum()
    n_above_50 = (obj_sigmoid > 0.50).sum()
    print(f"  boxes with sigmoid(obj) > 0.25 : {n_above_25}")
    print(f"  boxes with sigmoid(obj) > 0.50 : {n_above_50}")
    print()
    top5_sig = np.argsort(obj_sigmoid)[-5:][::-1]
    print("Top 5 by sigmoid(objectness):")
    for i in top5_sig:
        print(f"  idx={i}  raw_obj={objectness[i]:.4f}  sigmoid_obj={obj_sigmoid[i]:.4f}")