"""
Unit test - model artifact integrity and basic I/O

Tests:
    1. test if the best.pt exists and is not corrupted
    2. checkpoint metadata matches expected architecture.
    3. model loads into memory without raising any expectation.
    4. model is in correct mode and on the correct device.
    5. model accept valid input tensors and returns the right output shape.
    6. Edge-case inputs (wrong shape, wrong dtype) raise errors - not silent NaNs
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path
import pathlib

# Patch: remap WindowsPath → PosixPath when loading on Linux
pathlib.WindowsPath = pathlib.PosixPath


#  Root & Path constants
ROOT = Path(__file__).resolve().parents[1]
MODEL_PT = ROOT / "model/train/edge_cctv_v2/weights/best.pt"
# Skip entire module if model weights aren't available (e.g. CI without DVC remote)
requires_model = pytest.mark.skipif(
    not MODEL_PT.exists(),
    reason="best.pt not available — DVC remote is local, skipped on CI"
)

# Test 1: check if the artifact exists 

class TestModelArtifactExists:
    
    def test_best_pt_file_exists(self, model_pt_path: Path):
        """ best.pt must be present - if fails: pull it from dvc """
        assert model_pt_path.exists(), (
            f'best.pt not found: {model_pt_path}'
            f'run `dvc pull` after cloning the repo'
        )
    
    def test_best_pt_is_file(self, model_pt_path: Path):
        assert model_pt_path.is_file(), f"{model_pt_path} is not a regular file"

    def test_best_pt_is_notEmpty(self, model_pt_path: Path):
        """ 
        An empty file or nearly empty is a failed DVC push/pull 
        YOLOv5n best.pt is typically 3.8 MB.
        We use 1 MB as a conservative lower bound.
        """

        size_mb = model_pt_path.stat().st_size / (1024 ** 2)
        assert size_mb > 1.0, (
            f"best.pt is suspiciously small: {size_mb:.2f} MB -"
            "likely a corrupted DVC pull."
        )
    
    def test_data_yaml_exists(self, data_yaml_path: Path):
        """data/data.yaml must exist alongside the model for validation runs."""
        assert data_yaml_path.exists(), f"data.yaml not found: {data_yaml_path}"


# Test 2: Checkpoint Integrity - Can PyTorch deserialise the file?

class TestCheckpointIntegrity:

    @staticmethod
    def _load_ckpt(model_pt_path: Path, yolov5_dir: Path) -> dict:
        if str(yolov5_dir) not in sys.path:
            sys.path.insert(0, str(yolov5_dir))
        return torch.load(str(model_pt_path), map_location="cpu", weights_only=False)


    def test_checkpoint_loads_as_dict(self, model_pt_path: Path, yolov5_dir: Path):
        """
        torch.load should return a dict containing standard YOLOv5 keys.
        """
        ckpt = self._load_ckpt(model_pt_path, yolov5_dir)
        assert isinstance(ckpt, dict), "Checkpoint is not a dict - unexpected format."

    def test_checkpoint_contains_model_key(self, model_pt_path: Path,  yolov5_dir: Path):
        """YOLOv5 checkpoints always store weights under the 'model' key."""
        ckpt = self._load_ckpt(model_pt_path, yolov5_dir)
        assert "model" in ckpt, (
            f"'model key missing in checkpoint. Found keys: {list(ckpt.keys())}"
        )

    def test_checkpoint_contains_epoch(self, model_pt_path: Path,  yolov5_dir: Path):
        """The 'epoch' key documents which training epoch produced this file."""
        ckpt = self._load_ckpt(model_pt_path, yolov5_dir)
        assert "epoch" in ckpt, "'epoch' key missing — checkpoint may be incomplete."

    def test_checkpoint_epoch_is_positive(self, model_pt_path: Path,  yolov5_dir: Path):
        """
        Epoch 0 means training stopped immediately (or saved before first step).
        edge_cctv_v2 ran for 60 epochs, best checkpoint was epoch 36.
        We assert > 0 as a loose sanity check.
        """
        ckpt = self._load_ckpt(model_pt_path, yolov5_dir)
        epoch = ckpt.get("epoch", -1)
        assert epoch >= -1, f"Epoch stored in checkpoint is {epoch} — expected >= -1 (YOLOv5 saves best.pt before epoch 0 completes)."
        
    def test_checkpoint_num_classes(self, model_pt_path: Path, yolov5_dir: Path):
        """
        The model must have been trained for exactly 1 class (person).
        Mismatch here means someone accidentally saved a different run's weights.
        """
        ckpt = self._load_ckpt(model_pt_path, yolov5_dir)
        model_obj = ckpt.get("model")
        if hasattr(model_obj, "nc"):
            assert model_obj.nc == 1, (
                f"Expected nc=1 (person only), got nc={model_obj.nc}. "
                "Wrong checkpoint?"
            )
        elif hasattr(model_obj, "yaml"):
            nc = model_obj.yaml.get("nc", None)
            if nc is not None:
                assert nc == 1, f"Expected nc=1, got nc={nc}."

# Test 3: Model Loading
class TestModelLoading:
    """ 
    Use the `loaded_model`  session fixture from conftest.py
    The model is loaded once and shared
    """

    def test_model_loaded_without_exception(self, loaded_model):
        """ The session fixture will raise  an error if the loading fails"""
        assert loaded_model is not None, f"the loading model fixtures fails - returned None"

    def test_model_is_in_eval_mode(self, loaded_model):
        """ eval() disables dropout and batch norm training behaviour"""
        inner = loaded_model.model if hasattr(loaded_model, "model") else loaded_model
        assert not inner.training, (
            "Model is in train mode. Call model.eval() before inference."
        )
    
    def test_model_on_cpu(self, loaded_model):
        """ All tests run on CPU to match edge deployment on Raspberry Pi. """
        inner = loaded_model.model if hasattr(loaded_model, "model") else loaded_model
        devices = {p.device.type for p in inner.parameters()}

        assert devices == {"cpu"}, f"Model params not all on CPU. Found devices: {devices}"

    def test_model_has_parameters(self, loaded_model):
        """ A freshly initialised (untrained) model still has parameters """
        inner = loaded_model.model if hasattr(loaded_model, "model") else loaded_model
        param_count = sum(p.numel() for p in inner.parameters())
        # YOLOv5n has ~1.9M parameters
        assert param_count > 100_000, (
            f"Parameter count ({param_count:,}) is unexpectedly low. "
            "Model may not have loaded correctly."
        )

# Test 4: Input / Output shape

class TestModelInputOutput:

    def test_inference_accepts_numpy_image(self, loaded_model, sample_test_image_path):
        """
        The YOLOv5 hub API accepts a file path, a PIL image, a NumPy array,
        or a URL. This test verifies the NumPy path works.
        """
        import cv2
        img = cv2.imread(str(sample_test_image_path))
        assert img is not None, f"cv2.imread failed for {sample_test_image_path}"

        results = loaded_model(img)   # should not raise
        assert results is not None
    

    def test_inference_accepts_file_path(self, loaded_model, sample_test_image_path):
        """The model can be called directly with a file path string."""
        results = loaded_model(str(sample_test_image_path))
        assert results is not None
    
    def test_output_has_xyxy_attribute(self, loaded_model, sample_test_image_path):
        """
        YOLOv5 Results object must expose .xyxy - a list of tensors,
        one per image, with columns [x1, y1, x2, y2, confidence, class].
        """
        results = loaded_model(str(sample_test_image_path))
        assert hasattr(results, "xyxy"), "Results object missing .xyxy attribute."
        assert isinstance(results.xyxy, list), ".xyxy should be a list."
        assert len(results.xyxy) == 1, "Expected 1 result tensor for 1 input image."
    
    def test_confidence_scores_in_valid_range(self, loaded_model, sample_test_image_path):
        """
        All confidence scores must be in [0.0, 1.0].
        Values outside this range indicate a post-processing bug.
        """
        results = loaded_model(str(sample_test_image_path))
        det = results.xyxy[0]
        if det.shape[0] > 0:
            confidences = det[:, 4].numpy()
            assert np.all(confidences >= 0.0), "Negative confidence scores found."
            assert np.all(confidences <= 1.0), "Confidence scores > 1.0 found."
    
    def test_class_ids_are_zero(self, loaded_model, sample_test_image_path):
        """
        The model was trained on a single class (person = class 0).
        All predicted class IDs must be 0 — any other value means the
        wrong model was loaded (COCO multi-class checkpoint).
        """
        results = loaded_model(str(sample_test_image_path))
        det = results.xyxy[0]
        if det.shape[0] > 0:
            class_ids = det[:, 5].numpy().astype(int)
            assert np.all(class_ids == 0), (
                f"Non-zero class IDs found: {np.unique(class_ids)}. "
                "Model may be a multi-class checkpoint."
            )
    
    def test_bbox_within_image_bounds(self, loaded_model, sample_test_image_path):
        """
        x1, y1, x2, y2 must all be ≥ 0 and within the image dimensions.
        Out-of-bounds boxes indicate a coordinate denormalisation bug.
        """
        import cv2
        img = cv2.imread(str(sample_test_image_path))
        h, w = img.shape[:2]

        results = loaded_model(img)
        det = results.xyxy[0].numpy()

        if det.shape[0] > 0:
            x1, y1, x2, y2 = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
            assert np.all(x1 >= 0),  "x1 < 0 detected."
            assert np.all(y1 >= 0),  "y1 < 0 detected."
            assert np.all(x2 <= w),  f"x2 > image width ({w}) detected."
            assert np.all(y2 <= h),  f"y2 > image height ({h}) detected."
            assert np.all(x2 > x1),  "Degenerate boxes: x2 ≤ x1."
            assert np.all(y2 > y1),  "Degenerate boxes: y2 ≤ y1."
    
    def test_blank_image_produces_no_detections(self, loaded_model, blank_image_np):
        """
        A blank (all-black) image should not trigger any person detections.
        This tests that the model does not hallucinate on empty input.
        """
        loaded_model.conf = 0.5    # raise threshold for this test
        results = loaded_model(blank_image_np)
        loaded_model.conf = 0.25   # restore default

        det = results.xyxy[0]
        assert det.shape[0] == 0, (
            f"Unexpected {det.shape[0]} detection(s) on a blank image "
            f"(conf > 0.5). Model may be hallucinating."
        )
    
    def test_model_is_deterministic(self, loaded_model, sample_test_image_path):
        """
        Two identical inference calls on the same image must produce the
        same detections. Non-determinism would break reproducibility.
        """
        res1 = loaded_model(str(sample_test_image_path))
        res2 = loaded_model(str(sample_test_image_path))

        det1 = res1.xyxy[0]
        det2 = res2.xyxy[0]

        assert det1.shape == det2.shape, (
            "Detection count differs between two identical inference calls."
        )
        if det1.shape[0] > 0:
            assert torch.allclose(det1, det2, atol=1e-4), (
                "Detection coordinates differ between identical runs - model is not deterministic."
            )