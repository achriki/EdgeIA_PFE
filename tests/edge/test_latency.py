"""
Edge-Specific Tests — Raspberry Pi 3B deployment

These tests validate constraints that only matter on the target hardware: inference latency and model size on disk.  
Target hardware: Raspberry Pi 3B
    Architecture : ARM Cortex-A53 (armv7l / arm/v7)
    RAM          : 1 GB
    Runtime      : TFLite int8 via tflite-runtime

Test:
    1. TFLite model size <= 10 MB (fits in Pi RAM alongside the app)
    2. Single-frame inference ≤ 5s

NOTE: These tests are designed to run ON the Raspberry Pi, not on thedevelopment PC.
"""

import time
import pytest
import numpy as np
from pathlib import Path

#Edge hardware constraints
MAX_TFLITE_SIZE_MB = 10.0
MAX_LATENCY_S = 5.0

# helper - load TFLite interpreter

def get_interpreter(tflite_path: Path):
    """
    Load a TFLite interpreter using tflite-runtime (Pi) or tensorflow (dev PC).
    """

    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

    interpreter.allocate_tensors()
    return interpreter

# Test 1: model size

@pytest.mark.edge
def test_tflite_model_size_within_limit(model_tflite_path: Path):
    """
    The TFLite int8 model file must be ≤ 10 MB.
    """

    if not model_tflite_path.exists():
        pytest.skip(
            f"TFLite model not found at {model_tflite_path}. "
            "Run T6 (TFLite conversion) first."
        )

    size_mb = model_tflite_path.stat().st_size / (1024 ** 2)

    assert size_mb <= MAX_TFLITE_SIZE_MB, (
        f"TFLite model is {size_mb:.2f} MB — exceeds Pi limit of "
        f"{MAX_TFLITE_SIZE_MB} MB. "
        "Check that int8 quantisation ran correctly in T6."
    )

# Test 2: Inference Latency

@pytest.mark.edge
def test_tflite_inference_latency_within_limit(
    model_tflite_path: Path,
    sample_test_image_path: Path,
):
    """
    A single TFLite int8 inference call must complete in <= 5 seconds on the Raspberry Pi 3B.
    """
    if not model_tflite_path.exists():
        pytest.skip(
            f"TFLite model not found at {model_tflite_path}. "
            "Run T6 (TFLite conversion) first."
        )

    import cv2
    interpreter = get_interpreter(model_tflite_path)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input: resize to 640×640, normalise to [0, 255] int8
    img = cv2.imread(str(sample_test_image_path))
    img_resized = cv2.resize(img, (640, 640))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Warm-up run - not timed
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Timed runs
    N_RUNS = 3
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        times.append(time.perf_counter() - t0)

    avg_latency_s = sum(times) / N_RUNS

    assert avg_latency_s <= MAX_LATENCY_S, (
        f"Average TFLite inference latency = {avg_latency_s:.2f}s — "
        f"exceeds the T9 limit of {MAX_LATENCY_S}s on Pi 3B.\n"
        f"Individual runs: {[f'{t:.2f}s' for t in times]}\n"
        "Consider: verify int8 quantisation applied correctly, "
        "check no other processes are competing for Pi CPU."
    )