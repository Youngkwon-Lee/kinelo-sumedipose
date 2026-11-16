"""Synthetic-only regression tests for A4 frame-index and Vicon angle pilot."""
from pathlib import Path
import sys
import types

import pytest

from sumedipose import frame_joint_pilot
from sumedipose.frame_joint_pilot import ANCHORS, CAMERAS, MODELS, MODALITIES, aggregate_pilot


def _synthetic_rows(action="A4"):
    rows = []
    for subject in ANCHORS:
        for camera in CAMERAS:
            for model in MODELS:
                rows.append({
                    "subject": subject,
                    "camera": camera,
                    "model": f"pose_landmarker_{model}",
                    "action": action,
                    "unique_rgb_images_sampled": 24,
                    "poses_detected": 24,
                    "modalities": {
                        modality: {
                            "baseline_mae_deg": 14.0,
                            "baseline_mean_signed_bias_deg": -8.0,
                            "matched_zero_mae_deg": 13.5,
                            "best_vs_zero_in_sample_mae_delta_deg": 1.0,
                            "shift_zero_is_best_or_tied": False,
                            "matched_valid": 30,
                        }
                        for modality in MODALITIES
                    },
                })
    return rows


def test_A4_paired_aggregate_keeps_five_independent_calibrations():
    result = aggregate_pilot(_synthetic_rows(), action="A4")
    assert result["action"] == "A4 deadlift"
    assert result["inference_reports"] == 20
    assert result["image_inferences"] == 480
    assert result["independent_calibration_groups"] == 5
    assert result["new_independent_calibration_groups"] == 0
    assert len(result["aggregate_cells"]) == 8
    assert all(row["median_absolute_signed_bias_degrees"] == 8 for row in result["aggregate_cells"])
    assert result["safety"]["clinical_accuracy_validated"] is False
    assert result["safety"]["index_shift_is_time_sync_proof"] is not True if "index_shift_is_time_sync_proof" in result["safety"] else True


def test_existing_A3_default_unchanged():
    result = aggregate_pilot(_synthetic_rows("A3"))
    assert result["action"] == "A3 squat"
    assert result["new_independent_calibration_groups"] == 0


def test_reject_wrong_or_mixed_action():
    rows = _synthetic_rows("A4")
    rows[0]["action"] = "A3"
    with pytest.raises(ValueError, match="Mixed or unsupported"):
        aggregate_pilot(rows, action="A4")
    with pytest.raises(ValueError, match="Mixed or unsupported"):
        aggregate_pilot(_synthetic_rows("A4"), action="A5")


def test_inference_action_selection_reaches_reader_without_source_data(monkeypatch, tmp_path):
    # MediaPipe and cv2 placeholders, no real images or model weights used.
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "mediapipe", types.ModuleType("mediapipe"))
    seen = {}

    def synthetic_reader(root, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("reader-only")

    monkeypatch.setattr(frame_joint_pilot, "read_sample", synthetic_reader)
    with pytest.raises(RuntimeError, match="reader-only"):
        frame_joint_pilot.evaluate_recording(
            tmp_path, "S11", "C6", Path("synthetic.task"), action="A4")
    assert seen == {"camera": "C6", "subject": "S11", "action": "A4", "repeat": "D2"}


def test_unsupported_action_fails_before_accessing_any_raw_data(tmp_path):
    with pytest.raises(ValueError, match="Only the A3"):
        frame_joint_pilot.evaluate_recording(
            tmp_path, "S1", "C6", Path("synthetic.task"), action="A5")
