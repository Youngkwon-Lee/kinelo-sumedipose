"""Synthetic-only tests for paired action generalization; no participant data."""
import pytest
from sumedipose.action_generalization import analyse, ACTIONS
from sumedipose.calibration_strata import SESSIONS, CAMERAS, VARIANTS, MODALITIES


def synthetic_report(action, session, camera, variant, sha, mae):
    entry = {"valid": 24, "attempted": 24, "MAE_degrees": mae}
    return {
        "sample": f"{camera}{session}{action}D2",
        "model_variant": variant,
        "model_asset_sha256": sha,
        "unique_images_sampled": 24,
        "max_unique_images_requested": 24,
        "model_detected_poses": 24,
        "inference_mode": "IMAGE",
        "results": {
            modality: {"left": dict(entry), "right": dict(entry)}
            for modality in MODALITIES
        },
    }


def fixtures():
    sha = {"lite": "sha-lite", "full": "sha-full"}
    calibration = {
        "comparisons": [
            {"pair": "G65", "session": s, "sampson_median_px": i}
            for i, s in enumerate(SESSIONS, 1)
        ]
    }
    reports = {}
    for action in ACTIONS:
        for i, session in enumerate(SESSIONS, 1):
            for camera in CAMERAS:
                for variant in VARIANTS:
                    value = (10 + i) if action == "A3" else (11 + 2*i)
                    reports[action, session, camera, variant] = synthetic_report(
                        action, session, camera, variant, sha[variant], value)
    return calibration, reports, sha


def test_paired_actions_keep_five_independent_calibrations():
    calibration, reports, sha = fixtures()
    result = analyse(calibration, reports, sha)
    assert result["status"] == "complete_exploratory"
    assert result["paired_participants"] == 5
    assert result["independent_calibration_groups"] == 5
    assert result["expected_A4_model_image_inferences"] == 480
    assert len(result["comparisons"]) == 16
    assert len(result["within_participant_action_contrasts"]) == 8
    for row in result["within_participant_action_contrasts"]:
        assert row["paired_calibration_groups"] == 5
        assert row["A4_minus_A3_MAE_median_deg"] == 4
    assert result["raw_images_or_subject_level_pose_metrics_exported"] is False


def test_missing_action_report_is_not_silently_dropped():
    calibration, reports, sha = fixtures()
    del reports["A4", "S11", "C6", "full"]
    result = analyse(calibration, reports, sha)
    assert result["status"] == "incomplete_missing_reports"
    assert "A4D2_pose_S11_C6_full.json" in result["missing_model_reports"]


def test_model_weight_mismatch_rejected():
    calibration, reports, sha = fixtures()
    reports["A4", "S11", "C6", "full"]["model_asset_sha256"] = "unexpected"
    with pytest.raises(ValueError, match="model identity"):
        analyse(calibration, reports, sha)
