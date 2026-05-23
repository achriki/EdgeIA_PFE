"""
These tests are the formal pass/fail gate before deployment.
They assert that the model meets the project acceptance criteria defined in the PFE specification (mAP@0.5 ≥ 0.60) and that the TFLite int8 
quantisation does not degrade accuracy beyond the allowed margin (< 3 pts).

Tests:
    1. test best.pt (float32) - must meat the acceptance threshold
    2. test best.tflite (int 8) - must not regress more than 3 mAP points vs float 32
"""


import sys
import subprocess
import pytest
import numpy as np
from pathlib import Path

# Acceptance thresholds
MIN_MAP50 = 0.60
MAX_INT8_DROP = 0.03 #max allowed mAP loss after TFLite int8 quantisation

# golden baseline locked from edge_cctv_v2 val run
GOLDEN_MAP50 = 0.922

def run_val_and_get_map50(
        weights: Path,
        data_yaml: Path,
        yolov5_dir:Path,
        task: str = "test",

) -> float:
    """
    Run YOLOv5 val.py on the test split and parse the mAP@0.5 from stdout.
    Args:
        weights:    Path to .pt or .tflite weights file.
        data_yaml:  Path to data/dataset.yaml.
        yolov5_dir: Path to the cloned YOLOv5 repository.
        task:       'test' evaluates on the test split (not val).

    Returns:
        mAP@0.5 as a float in [0, 1].
    """

    cmd = [
        sys.executable, str(yolov5_dir / "val.py"),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--img", "640",
        "--conf", "0.25",
        "--iou", "0.45",
        "--task", task,
        "--device", "cpu",
        "--workers", "0",
        "--exist-ok",
        "--verbose"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(yolov5_dir)
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"val.py failed with return code: {result.returncode}\n"
            f"stderr: \n {result.stderr[-2000:]}"
        )
    
    #val.py summary line is like: "all <image> <labels> <P> <R> <mAP50> <mAP50-95>"

    for line in result.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("all"):
            parts = stripped.split()
            #parts = ["all", image, labels, P, R, mAP50, mAP50-95]
            if len(parts) >= 6:
                return float(parts[5])
    
    raise RuntimeError (
        "Could not parse mAP@0.5 from val.py output.\n"
        f"  Full Stdout:\n{result.stdout}\nFull stderr:\n{result.stderr}"
    )

# test 1: Float32 model - must meet the acceptance threshold

class TestFloat32Performance:
    def test_map50_meets_acceptance_threshold(
            self,
            model_pt_path: Path,
            data_yaml_path: Path,
            yolov5_dir: Path,
            golden_metrics: Path
    ):
        """
        best.pt must achieve mAP@0.5 >= 0.60 on test set.
        0.60 is the minimum defined in the PFE specification for deployment approval.
        """

        map50 = run_val_and_get_map50(
            weights=model_pt_path,
            data_yaml=data_yaml_path,
            yolov5_dir=yolov5_dir
        )

        assert map50 >= MIN_MAP50, (
            f"best.pt mAP@0.5 = {map50:.4f} - below acceptance threshold "
            f"of {MIN_MAP50}. Model must not be deployed."
        )

        # Informational regression check
        drop = golden_metrics["map50"] - map50
        if drop > 0.2:
            pytest.warns(
                UserWarning,
                match=f"mAP dropped {drop:.3f} vs golden baseline "
                      f"({golden_metrics['map50']}). Investigate before deploying."
            )

# test 2: TFLite int8 model - must not regress more than 3 mAP points

class TestTFLiteIntPerformance:

    def test_int8_map50_within_allowed_degradation(
            self,
            model_pt_path: Path,
            model_tflite_path: Path,
            data_yaml_path: Path,
            yolov5_dir: Path
    ):
        """
        The TFLite int8 model (T6 output) must not lose more than 3 mAP points compared to the float32 baseline.
        """

        if not model_tflite_path.exists():
            pytest.skip(
                f"TFLite model not found at {model_tflite_path}. "
                "Run T6 (TFLite conversion notebook) first."
            )
        
        map50_float32 = run_val_and_get_map50(
            weights=model_pt_path,
            data_yaml=data_yaml_path,
            yolov5_dir=yolov5_dir,
        )

        map50_int8 = run_val_and_get_map50(
            weights=model_tflite_path,
            data_yaml=data_yaml_path,
            yolov5_dir=yolov5_dir,
        )

        drop = map50_float32 - map50_int8

        assert drop <= MAX_INT8_DROP, (
            f"TFLite int8 mAP@0.5 = {map50_int8:.4f} — dropped {drop:.4f} "
            f"vs float32 ({map50_float32:.4f}). "
            f"Allowed maximum drop = {MAX_INT8_DROP}. "
            "Re-check the representative dataset used during quantisation"
        )