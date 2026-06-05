"""
conftest.py - Shared fixtures for the EdgeIA PFE pytest suite.

Scope hierarchy:
  session: loaded once for the entire test run
  module: loaded once per test file
  function: loaded fresh for each test

Project root: C:/Users/ASUS/Desktop/EdgeIA_PFE/
"""
import os
import time
import pytest
import numpy as np
from pathlib import Path
import pathlib
from PIL import Image

# Patch: remap WindowsPath → PosixPath when loading on Linux
pathlib.WindowsPath = pathlib.PosixPath

#  Root & Path constants
ROOT = Path(__file__).resolve().parents[1]

MODEL_PT = ROOT / "model/train/edge_cctv_v2/weights/best.pt"
MODEL_TFLite = ROOT / "model/train/edge_cctv_v2/best_int8.tflite"  # model compression output
DATA_YAML = ROOT / "data/data.yaml"
TEST_IMGS = ROOT / "data/splits/test/images"
TEST_LBLS = ROOT / "data/splits/test/labels"
YOLOV5_DIR = ROOT / "model/yolov5"

#Golden baseline from edge_cctv_v2 (float31 val run)
GOLDEN_MAP50 = 0.922   # mAP@0.5
GOLDEN_PRECISION = 0.859
GOLDEN_RECALL = 0.904

# Acceptable thresholds
MIN_MAP50 = 0.60
MAX_INT8_DROP = 0.03 # max allowed mAP drop after TFLite int 3 quantisation
MAX_LATENCY_MS = 500 #single-image CPU inference ceilling



# Path fixtures
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the EdgeIA_PFE project root."""
    return ROOT


@pytest.fixture(scope="session")
def model_pt_path() -> Path:
    """Path to edge_cctv_v2 best.pt (float32)."""
    return MODEL_PT


@pytest.fixture(scope="session")
def model_tflite_path() -> Path:
    """Path to TFLite int8 export (T4 output)."""
    return MODEL_TFLite


@pytest.fixture(scope="session")
def data_yaml_path() -> Path:
    """Path to data/data.yaml."""
    return DATA_YAML


@pytest.fixture(scope="session")
def test_images_dir() -> Path:
    """Path to data/test/images/."""
    return TEST_IMGS


@pytest.fixture(scope="session")
def test_labels_dir() -> Path:
    """Path to data/test/labels/."""
    return TEST_LBLS


@pytest.fixture(scope="session")
def yolov5_dir() -> Path:
    """Path to the cloned YOLOv5 repository."""
    return YOLOV5_DIR

# Model fixture - load once per session

@pytest.fixture(scope="session")
def loaded_model():
    import torch                      # lazy — only runs when fixture is called
    import sys
    sys.path.insert(0, str(YOLOV5_DIR))

    assert MODEL_PT.exists(), (
        f"best.pt not found at {MODEL_PT}.\n"
        f"Run DVC pull or check model/train/edge_cctv_v2/weights/"
    )

    model = torch.hub.load(
        str(YOLOV5_DIR),
        "custom",
        path=str(MODEL_PT),
        source="local",
        force_reload=False,
        verbose=False,
    )

    model.eval()
    model.cpu()
    return model

# Image fixtures
@pytest.fixture(scope="session")
def sample_test_image_path() -> Path:
    images = sorted(TEST_IMGS.glob("*.jpg"))
    assert len(images) > 0 , f"No images found in {TEST_IMGS}"
    return images[0]

@pytest.fixture(scope="session")
def all_test_images_paths()->list:
    images = sorted(TEST_IMGS.glob("*.jpg"))
    assert len(images) > 0, f"No images found in {TEST_IMGS}"
    return images

@pytest.fixture(scope="function")
def blank_image_np() -> np.ndarray:
    """
    A 640×640 blank (all-zero) BGR NumPy image.
    Used to test that the model produces zero detections on empty input.
    """
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture(scope="function")
def random_noise_image_np() -> np.ndarray:
    """
    A 640×640 random noise BGR NumPy image.
    Used to stress-test that output shapes are always valid.
    """
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)

# Golden baseline constants
@pytest.fixture(scope="session")
def golden_metrics() -> dict:
    """
    model reference metrics locked after evaluation.
    Any regression below these values must be caught and reported.
    """
    return {
        "map50": GOLDEN_MAP50,
        "precision": GOLDEN_PRECISION,
        "recall": GOLDEN_RECALL,
    }


@pytest.fixture(scope="session")
def thresholds() -> dict:
    """
    Acceptance thresholds defined in the project requirements.
    These are the pass/fail criteria.
    """
    return {
        "min_map50": MIN_MAP50,
        "max_int8_drop": MAX_INT8_DROP,
        "max_latency_ms": MAX_LATENCY_MS,
    }

# Utility fixtures

@pytest.fixture(scope="function")
def timer():
    """
    Simple wall-clock timer fixture.

    Usage inside a test function
    """
    class _Timer:
        def __init__(self):
            self._start = None

        def start(self):
            self._start = time.perf_counter()

        def stop(self) -> float:
            assert self._start is not None, "Call timer.start() first."
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._start = None
            return elapsed_ms
    return _Timer()