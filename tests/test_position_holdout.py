"""Synthetic-only tests for strict source matching and held-out 3D shape metric."""
import numpy as np
import pytest

from sumedipose.position_holdout import (
    ANCHORS, CAMERAS, VARIANTS, aggregate, fit_fixed_similarity,
    heldout_discrepancy, root_relative, strict_unique_pairs,
)


def test_dropped_frame_placeholder_is_never_selected_as_reference():
    records = [
        {"path": "C1S1A3D2_0001.jpg", "factor": 2, "dup_num": 1},
        {"path": "C1S1A3D2_0001.jpg", "factor": 2, "dup_num": 1},
        {"path": "C1S1A3D2_0001.jpg", "factor": 2, "dup_num": 2},
        {"path": "C1S1A3D2_0002.jpg", "factor": 1, "dup_num": 1},
    ]
    picks, audit = strict_unique_pairs(
        records, ["C1S1A3D2_0001.jpg", "C1S1A3D2_0002.jpg"], count=2)
    assert [i for _, i in picks] == [2, 3]
    assert audit["previous_index_selection_invalid_pairs"] == 1
    assert audit["unique_verified_jpgs_selected"] == 2


def test_without_verified_marker_or_unique_images_fail_closed():
    records = [
        {"path": "a.jpg", "factor": 2, "dup_num": 1},
        {"path": "b.jpg", "factor": 1, "dup_num": 1},
    ]
    with pytest.raises(ValueError, match="Insufficient"):
        strict_unique_pairs(records, ["a.jpg", "b.jpg"], count=2)
    with pytest.raises(ValueError, match="count"):
        strict_unique_pairs(records, ["a.jpg", "b.jpg"], count=1)


def _geometry():
    return np.asarray([
        [-0.08, 0.05, 0.04], [-0.12, -0.37, 0.01], [-0.09, -0.75, 0.06],
        [0.08, -0.05, -0.04], [0.12, -0.43, -0.01], [0.09, -0.81, -0.06],
    ])


def test_fixed_rotation_scale_fitted_on_train_only_and_generalizes():
    x = np.stack([root_relative(_geometry() + np.array(
        [0.00, i / 500, ((i % 3) - 1) / 100])) for i in range(8)])
    rot = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    y = x @ rot * 1120
    fit_r, scale = fit_fixed_similarity(x[:6], y[:6])
    assert scale == pytest.approx(1120, rel=1e-10)
    assert np.linalg.det(fit_r) == pytest.approx(1)
    heldout = heldout_discrepancy(
        [(i, x[i], y[i]) for i in (6, 7, 6, 7, 6, 7)], fit_r, scale)
    assert heldout["mean_discrepancy_pct_reference_femur_length"] == 0
    assert heldout["physical_mpjpe_validated"] is False
    assert heldout["absolute_root_translation_measured"] is False


def test_refuse_training_less_than_six_frames_and_degenerate():
    p = np.zeros((5, 6, 3))
    with pytest.raises(ValueError, match=">=6"):
        fit_fixed_similarity(p, p)
    with pytest.raises(ValueError, match="Degenerate"):
        fit_fixed_similarity(np.zeros((6, 6, 3)), np.zeros((6, 6, 3)))


def _synthetic_records():
    out = []
    for subj in ANCHORS:
        for camera in CAMERAS:
            for model in VARIANTS:
                m = {
                    "mean_discrepancy_pct_reference_femur_length": 16.0,
                    "valid_evaluation_frames": 11,
                }
                sample = {
                    "model_images_attempted": 24,
                    "pose_detections": 23,
                    "previous_index_selection_invalid_pairs": 1,
                }
                out.append({
                    "subject": subj, "camera": camera, "model": model,
                    "A3_heldout": dict(m), "A4_cross_action": dict(m),
                    "source_alignment_audit": {
                        "A3": dict(sample), "A4": dict(sample)
                    },
                })
    return out


def test_equal_calibration_weighting_and_no_inflated_independence():
    out = aggregate(_synthetic_records())
    assert out["independent_calibration_groups"] == 5
    assert out["new_independent_calibration_groups"] == 0
    assert out["total_recording_model_pairs"] == 40
    assert out["total_model_images_attempted"] == 960
    assert out["previous_sampling_invalid_coordinate_pairs"] == 20
    assert len(out["cells"]) == 8
    assert out["not_absolute_mpjpe_or_validated_mm_accuracy"] is True
    assert out["physical_sync_verified"] is False
    assert out["no_training_on_A4"] is True


def test_missing_anchor_in_aggregate_is_rejected():
    rec = _synthetic_records()
    rec[-1]["subject"] = "S1"
    with pytest.raises(ValueError):
        aggregate(rec)
