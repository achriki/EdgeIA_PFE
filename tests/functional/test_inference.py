"""
Functional tests - test the model inference correctness

Scope: validate that the model produces correct, pipeline-safe detections
on real CCTV test images.

Project thresholds:
    CONF_THRESHOLD = 0.25 (YOLOv5 default, used in training eval)
    IOU_THRESHOLD = 0.45 (NMS threshold used during training)
"""

import sys
import math
import pytest
import torch
import numpy as np
import cv2
from pathlib import Path

# Project thresholds
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Based on dataset EDA: 6273 annotations over ~753 images → ~8.3 per image.
# The 76-image test split (~10%) carries roughly 600+ ground-truth boxes.
# We use a very conservative floor (50) to avoid flaky failures on edge cases.
MIN_TOTAL_DETECTIONS_TESTSET = 50


MIN_DETECTIONS_POSITIVE_IMAGE = 1

# compute pairwise IoU for a set of boxes

def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the N×N IoU matrix for N boxes in xyxy format.

    Args:
        boxes: np.ndarray of shape [N, 4] with columns [x1, y1, x2, y2]

    Returns:
        iou_matrix: np.ndarray of shape [N, N], diagonal is 1.0
    """
    n = len(boxes)
    iou_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i, n):
            x1 = max(boxes[i, 0], boxes[j, 0])
            y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2])
            y2 = min(boxes[i, 3], boxes[j, 3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            union = area_i + area_j - intersection

            iou = intersection / union if union > 0 else 0.0
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    return iou_matrix

# Test 1: model detection presence
class TestDetectionPresence:
    """
    Verify the model produces detections on real images from the test set.
    """
    def test_detects_person_in_first_test_image(
        self, loaded_model, sample_test_image_path: Path
    ):
        """
        The very first test image must yield at least one detection.
        """
        results = loaded_model(str(sample_test_image_path))
        det = results.xyxy[0]
        assert det.shape[0] >= MIN_DETECTIONS_POSITIVE_IMAGE, (
            f"Zero detections on {sample_test_image_path.name}.\n"
            f"Expected at least {MIN_DETECTIONS_POSITIVE_IMAGE} person(s).\n"
            "Check that CONF_THRESHOLD=0.25 is set on the model."
        )

    def test_detects_across_full_test_set(
        self, loaded_model, all_test_images_paths: list 
    ):
        """
        Run inference over all 76 test images and accumulate total detections.
        The sum must exceed MIN_TOTAL_DETECTIONS_TESTSET (conservative floor).

        This catches catastrophic weight corruption where the model loads
        without error but produces near-zero detections on everything.
        """
        loaded_model.conf = CONF_THRESHOLD
        loaded_model.iou  = IOU_THRESHOLD

        total_detections = 0
        images_with_zero = []

        for img_path in all_test_images_paths:
            results = loaded_model(str(img_path))
            n = results.xyxy[0].shape[0]
            total_detections += n
            if n == 0:
                images_with_zero.append(img_path.name)

        assert total_detections >= MIN_TOTAL_DETECTIONS_TESTSET, (
            f"Only {total_detections} detections across {len(all_test_images_paths)} "
            f"test images (floor = {MIN_TOTAL_DETECTIONS_TESTSET}).\n"
            f"Images with zero detections: {images_with_zero}"
        )

    def test_majority_of_test_images_have_detections(
        self, loaded_model, all_test_images_paths: list
    ):
        """
        At least 70% of test images must have ≥1 detection.

        Our dataset EDA showed people appear in virtually every frame.
        A model that misses more than 30% of frames is not acceptable
        for a people-counting / CCTV monitoring pipeline.
        """
        loaded_model.conf = CONF_THRESHOLD
        loaded_model.iou  = IOU_THRESHOLD

        detected = sum(
            1 for p in all_test_images_paths
            if loaded_model(str(p)).xyxy[0].shape[0] > 0
        )
        ratio = detected / len(all_test_images_paths)

        assert ratio >= 0.70, (
            f"Only {detected}/{len(all_test_images_paths)} images "
            f"({ratio:.1%}) had ≥1 detection. Expected ≥70%."
        )

# Test 2: Absence detection 
class TestDetectionAbsence:
    """
    Verify the model does not hallucinate people on frames that have one.
    False positives are especially costly in a people-counting system they corrupt occupancy estimates and trigger false alarms.
    """
    
    def test_blank_black_image_no_detections(
        self, loaded_model, blank_image_np: np.ndarray
    ):
        """
        A fully black 640×640 image must yield zero detections at conf=0.5.
        We use a higher threshold (0.5) than the default because even random
        noise can occasionally produce very low-confidence (<0.3) detections.
        """
        loaded_model.conf = 0.50
        results = loaded_model(blank_image_np)
        loaded_model.conf = CONF_THRESHOLD   # restore

        det = results.xyxy[0]
        assert det.shape[0] == 0, (
            f"Model produced {det.shape[0]} detection(s) on a blank image "
            f"at conf=0.50. Confidences: {det[:, 4].tolist()}"
        )

    def test_random_noise_image_low_confidence(
        self, loaded_model, random_noise_image_np: np.ndarray
    ):
        """
        A random noise image should produce either zero detections or
        only very low-confidence detections (< 0.50) that NMS would
        discard in a real pipeline.
        """
        loaded_model.conf = 0.50
        results = loaded_model(random_noise_image_np)
        loaded_model.conf = CONF_THRESHOLD   # restore

        det = results.xyxy[0]
        if det.shape[0] > 0:
            max_conf = det[:, 4].max().item()
            assert max_conf < 0.50, (
                f"Random noise image produced a detection with "
                f"confidence {max_conf:.3f} ≥ 0.50."
            )

    def test_solid_grey_image_no_detections(self, loaded_model):
        """
        Solid grey images (common when a CCTV feed drops or is overexposed)
        must not produce detections.
        """
        grey = np.full((640, 640, 3), 128, dtype=np.uint8)

        loaded_model.conf = 0.50
        results = loaded_model(grey)
        loaded_model.conf = CONF_THRESHOLD

        det = results.xyxy[0]
        assert det.shape[0] == 0, (
            f"Solid grey image produced {det.shape[0]} detection(s)."
        )

# Test 3: Output validity
class TestOutputValidity:
    """
    Validate the structure and value ranges of every detection produced
    across all test images. One bad value in one field fails the test.

    Detection tensor columns: [x1, y1, x2, y2, confidence, class_id]
    """
    
    @pytest.fixture(scope="class")
    def all_detections(self, loaded_model, all_test_images_paths):
        """
        Run inference on every test image and stack all detection rows.
        Scoped to class so this expensive pass runs only once.
        """
        loaded_model.conf = CONF_THRESHOLD
        loaded_model.iou  = IOU_THRESHOLD

        all_dets = []
        all_imgs = []
        for img_path in all_test_images_paths:
            img = cv2.imread(str(img_path))
            results = loaded_model(img)
            det = results.xyxy[0].numpy()
            if det.shape[0] > 0:
                all_dets.append(det)
                h, w = img.shape[:2]
                all_imgs.extend([(w, h)] * det.shape[0])

        return (
            np.vstack(all_dets) if all_dets else np.empty((0, 6)),
            all_imgs,
        )

    def test_detection_tensor_has_six_columns(
        self, loaded_model, sample_test_image_path
    ):
        """Output tensor must always have exactly 6 columns."""
        results = loaded_model(str(sample_test_image_path))
        det = results.xyxy[0]
        assert det.shape[1] == 6, (
            f"Expected 6 columns [x1,y1,x2,y2,conf,cls], got {det.shape[1]}."
        )

    def test_all_confidences_above_threshold(self, all_detections):
        """
        Every surviving detection must have confidence ≥ CONF_THRESHOLD (0.25).
        Detections below the threshold should have been suppressed by NMS.
        """
        dets, _ = all_detections
        if dets.shape[0] == 0:
            pytest.skip("No detections to validate.")
        confs = dets[:, 4]
        below = np.sum(confs < CONF_THRESHOLD)
        assert below == 0, (
            f"{below} detection(s) have confidence < {CONF_THRESHOLD}. "
            "NMS threshold may not be applied correctly."
        )

    def test_all_class_ids_are_zero(self, all_detections):
        """
        The model was trained on a single class (person = 0).
        Any non-zero class ID means the wrong checkpoint was loaded.
        """
        dets, _ = all_detections
        if dets.shape[0] == 0:
            pytest.skip("No detections to validate.")
        class_ids = dets[:, 5].astype(int)
        non_zero = np.unique(class_ids[class_ids != 0])
        assert len(non_zero) == 0, (
            f"Non-zero class IDs found: {non_zero}. "
            "Model is not single-class or wrong checkpoint loaded."
        )
        
    def test_no_nan_or_inf_in_output(self, all_detections):
        """
        NaN or Inf values in any detection field will silently corrupt
        downstream calculations (IoU, centroid tracking, counting).
        """
        dets, _ = all_detections
        if dets.shape[0] == 0:
            pytest.skip("No detections to validate.")
        assert not np.any(np.isnan(dets)), "NaN values found in detection tensor."
        assert not np.any(np.isinf(dets)), "Inf values found in detection tensor."

# Test 4: INFERENCE CONSISTENCY
# Does the model produce the same result when run twice on the same frame?

class TestInferenceConsistency:
    """
    Reproducibility is required for the MLOps pipeline: DVC stages must
    produce identical outputs on identical inputs. Non-determinism would
    make experiment tracking with MLflow meaningless.
    """

    def test_same_image_same_result_twice(
        self, loaded_model, sample_test_image_path: Path
    ):
        """Two consecutive inferences on the same path must be identical."""
        res1 = loaded_model(str(sample_test_image_path))
        res2 = loaded_model(str(sample_test_image_path))

        det1 = res1.xyxy[0]
        det2 = res2.xyxy[0]

        assert det1.shape == det2.shape, (
            f"Detection count changed between calls: "
            f"{det1.shape[0]} vs {det2.shape[0]}."
        )
        if det1.shape[0] > 0:
            assert torch.allclose(det1, det2, atol=1e-5), (
                "Detection values differ between identical inference calls."
            )

    def test_same_image_same_result_ten_times(
        self, loaded_model, sample_test_image_path: Path
    ):
        """
        Run inference 10 times and verify all results are identical.
        This catches non-determinism introduced by threading or
        floating-point rounding differences across runs.
        """
        first = loaded_model(str(sample_test_image_path)).xyxy[0]

        for i in range(9):
            current = loaded_model(str(sample_test_image_path)).xyxy[0]
            assert first.shape == current.shape, (
                f"Run {i+2}: detection count changed ({first.shape[0]} → "
                f"{current.shape[0]})."
            )
            if first.shape[0] > 0:
                assert torch.allclose(first, current, atol=1e-5), (
                    f"Run {i+2}: detection values differ from first run."
                )
