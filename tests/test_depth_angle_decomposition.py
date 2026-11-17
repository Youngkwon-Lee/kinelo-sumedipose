"""Synthetic-only tests for the 3D knee included-angle decomposition."""
from __future__ import annotations
import math

import numpy as np
import pytest

from sumedipose.depth_angle_decomposition import (
    ANCHORS, CAMERAS, FIELDS, GROUPS, MODELS, PHASES,
    _mean_metrics, angle_terms, grouped_summary, phase_bin,
)


def _points(depth=0.0):
    return np.asarray([[-1., 0., 0.], [0., 0., 0.], [1., 0., depth]])


def test_z_component_exact_additive_angle_identity():
    ref = _points(depth=0)
    pred = _points(depth=1)
    p2 = ref[:, :2]
    got = angle_terms(pred, ref, ref, p2, p2)
    assert got["signed_flat_xy_error"] == pytest.approx(0)
    assert got["signed_3d_error_cam"] == pytest.approx(-45)
    assert got["signed_z_component_gap"] == pytest.approx(-45)
    assert got["abs_3d_error_wcs"] == pytest.approx(45)
    assert got["abs_segment_ratio_log_gap"] > 0
    assert (got["signed_flat_xy_error"] + got["signed_z_component_gap"]
            == pytest.approx(got["signed_3d_error_cam"]))


def test_reference_z_term_mismatch_not_implied_real_exposure():
    ref = _points(depth=1)
    pred = _points(depth=0)
    got = angle_terms(pred, ref, ref, pred[:, :2], ref[:, :2])
    assert got["signed_z_component_gap"] == pytest.approx(45)
    assert got["signed_flat_xy_error"] == pytest.approx(0)
    assert got["reference_z_angle_effect"] == pytest.approx(-45)


def test_model_plane_offset_without_depth():
    ref = np.asarray([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.]])
    pred = np.asarray([[-1.,0.,0.],[0.,0.,0.],[1.,1.,0.]])
    got = angle_terms(pred, ref, ref, pred[:, :2], ref[:, :2])
    assert got["signed_z_component_gap"] == pytest.approx(0)
    assert abs(got["signed_flat_xy_error"]) > 1


def test_camera_and_wcs_reference_must_agree():
    with pytest.raises(ValueError, match="deviates"):
        angle_terms(_points(0), _points(0), _points(1),
                    _points(0)[:, :2], _points(0)[:, :2])


def test_angle_phase_bins_not_clinical():
    assert phase_bin(119.99) == PHASES[0]
    assert phase_bin(120.) == PHASES[1]
    assert phase_bin(149.99) == PHASES[1]
    assert phase_bin(150.) == PHASES[2]


def _synthetic_row(sub, camera, model):
    metrics = {"valid_joint_angles": 30}
    metrics.update({field: 2.0 for field in FIELDS})
    # Distinguish one group to confirm each group receives equal weight
    metrics["signed_3d_error_cam"] = 10. if sub == "S1" else 2.
    metrics["signed_flat_xy_error"] = 2.
    metrics["signed_z_component_gap"] = 8. if sub == "S1" else 0.
    return {
        "subject": sub, "calibration_group": GROUPS[sub],
        "camera": camera, "model": model,
        "unique_images": 24, "pose_detections": 24,
        "all": metrics, "by_phase": {phase:metrics for phase in PHASES},
    }


def test_grouped_summary_five_anchors_and_no_pseudoreplication():
    rows = [_synthetic_row(s,c,m) for s in ANCHORS for c in CAMERAS for m in MODELS]
    out = grouped_summary(rows, "anchors")
    assert out["n_subjects"] == 5
    assert out["n_calibration_groups"] == 5
    assert out["model_reports"] == 20
    assert out["model_image_inferences"] == 480
    cell = next(z for z in out["results"] if z["camera"]=="C6"
                and z["model"]=="full" and z["phase"]=="all")
    assert cell["signed_3d_error_cam_mean_of_group_means"] == pytest.approx(3.6)
    assert cell["signed_flat_xy_error_mean_of_group_means"] == pytest.approx(2)
    assert cell["signed_z_component_gap_mean_of_group_means"] == pytest.approx(1.6)


def test_26_census_group_count_and_missing_subjects():
    assert len(GROUPS) == 26
    assert {"S5","S10"}.isdisjoint(GROUPS)
    assert [sum(g==a for g in GROUPS.values()) for a in ANCHORS] == [4,4,6,6,6]
    rows = [_synthetic_row(s,c,m) for s in GROUPS for c in CAMERAS for m in MODELS]
    result = grouped_summary(rows, "available26")
    assert result["n_subjects"] == 26 and result["model_reports"] == 104
    assert result["model_image_inferences"] == 2496


def test_incomplete_cohort_is_rejected():
    rows = [_synthetic_row(s,c,m) for s in ANCHORS for c in CAMERAS for m in MODELS]
    with pytest.raises(ValueError, match="Incomplete"):
        grouped_summary(rows[:-1], "anchors")


def test_empty_phase_has_no_metrics():
    out = _mean_metrics([])
    assert out["valid_joint_angles"] == 0
    assert all(out[k] is None for k in FIELDS)

def test_source_valid_report_is_required(tmp_path):
    """Do not silently fall back to old midpoint-index reference labels."""
    import json
    from sumedipose.depth_angle_decomposition import _strict_report
    directory = tmp_path / "strict_source_pairing_20261117"
    directory.mkdir()
    report_path = directory / "strict_A3D2_S1_C1_lite.json"
    with pytest.raises(ValueError, match="Missing"):
        _strict_report(tmp_path, "S1", "C1", "lite")
    report_path.write_text(json.dumps({"pairing": {
        "new_reference_uses_dup_num_equals_factor": False
    }}), encoding="utf8")
    with pytest.raises(ValueError, match="source-valid"):
        _strict_report(tmp_path, "S1", "C1", "lite")
    report_path.write_text(json.dumps({"pairing": {
        "new_reference_uses_dup_num_equals_factor": True
    }}), encoding="utf8")
    assert _strict_report(tmp_path, "S1", "C1", "lite")["pairing"][
        "new_reference_uses_dup_num_equals_factor"
    ]
