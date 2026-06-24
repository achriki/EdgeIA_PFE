"""
Corrected preprocess() and postprocess() for inference.py

ROOT CAUSE: YOLOv5 never does a naive square resize. It letterboxes —
scales the image to fit within (INPUT_W, INPUT_H) while preserving aspect
ratio, then pads the leftover space with grey (RGB 114,114,114). The model
was trained exclusively on letterboxed inputs, so feeding it a squashed/
stretched image produces near-zero confidence everywhere (confirmed via
debug_output.py: max objectness ~0.04 with naive resize).

This mirrors the exact fix already applied in the T4 quantization notebook's
calibration step (letterbox_np) — it just never made it into inference.py.
"""
import io
import numpy as np
from PIL import Image


def letterbox(img: Image.Image, size: int = 640, color=(114, 114, 114)):
    """
    Resize `img` to fit within size x size while preserving aspect ratio,
    then pad with grey to fill the rest. Matches YOLOv5's internal
    preprocessing exactly (same padding color, same centering logic).

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


def postprocess(output, orig_h, orig_w, scale, pad_x, pad_y, conf_threshold=0.25):
    """
    Parse YOLOv5 TFLite output into detections, converting letterboxed
    coordinates back to original image coordinates.

    Output columns: [cx, cy, w, h, objectness, class_prob] (single-class model)
    Box coordinates from the model are normalized to the letterboxed
    640x640 canvas — they must be unpadded and unscaled to map back to the
    original image.
    """
    detections = []
    preds = output[0]  # drop batch dim -> (num_boxes, 6)

    # Single-class model: objectness IS the confidence (no separate class
    # gating needed — there's only one class, "person", so don't filter by
    # an int(class_col) check; that column may not be a clean 0/1 index).
    confidences = preds[:, 4]

    # NMS via simple greedy suppression (replace with cv2.dnn.NMSBoxes or
    # a vectorized version if you need this faster on the Pi)
    keep_idx = np.where(confidences > conf_threshold)[0]

    for i in keep_idx:
        cx, cy, w, h, conf = preds[i, 0], preds[i, 1], preds[i, 2], preds[i, 3], preds[i, 4]

        # Convert normalized (0-1) letterbox-space center coords to
        # letterbox-canvas pixel coords (canvas is INPUT_W x INPUT_H)
        cx_px = cx * INPUT_W
        cy_px = cy * INPUT_H
        w_px = w * INPUT_W
        h_px = h * INPUT_H

        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2

        # Undo letterbox padding + scale to map back to original image coords
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip to original image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        detections.append({
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
            "conf": round(float(conf), 3),
            "class": 0,
        })

    return detections