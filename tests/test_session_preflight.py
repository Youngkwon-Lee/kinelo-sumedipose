"""Synthetic-only preflight tests. No study participant or original data."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from sumedipose.session_preflight import audit_session, main


TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "independent-calibration-session-template.json"


def _blank():
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _synthetic_ready():
    record = deepcopy(_blank())
    record["research_status"] = "collected_for_review"
    record["session_code"] = "CAL-SYNTHETIC-001"
    record["acquisition_date"] = "2026-11-17"
    record["equipment"] = {name: True for name in record["equipment"]}
    rig = record["camera_setup"]
    rig.update(site_code="SYNTHETIC_SITE", rig_config_version="RIG_TEST_V1",
               new_calibration_acquisition=True,
               new_setup_or_remount_evidence="SYNTHETIC_ONLY_REVIEW_REFERENCE",
               metric_compatible_with_SUMediPose_G65="VERIFIED")
    calib = record["calibration"]
    calib.update(square_size_mm=60.0,
                 fit_holdout_disjoint_verified=True,
                 intrinsics_frozen_before_holdout=True,
                 extrinsics_frozen_before_holdout=True,
                 C6_C5_fit_frames_count=6,
                 C6_C5_heldout_frames_count=40,
                 heldout_board_quality_metric={
                     "name": "C6_C5_Sampson_median_px", "value_px": 2.5
                 })
    record["sync"].update(
        camera_exposure_timestamps_observed=True,
        Vicon_reference_clock_verified=True,
        common_trigger_or_independent_event_verified=True,
        physical_timestamp_alignment_status="EVIDENCE_REVIEWED",
    )
    record["reference"] = {
        "camera_Vicon_registration_verified": True,
        "anatomical_joint_definition_reviewed": True,
    }
    record["permissions"] = {name: True for name in record["permissions"]}
    return record


def test_public_blank_template_blocks_and_does_not_increment_sessions():
    result = audit_session(_blank())
    assert result["status"] == "blocked"
    assert "equipment_availability_confirmed" in result["unmet_gates"]
    assert "physical_time_reference_reviewed" in result["unmet_gates"]
    assert "permissions_confirmed" in result["unmet_gates"]
    assert result["new_independent_calibration_sessions_added"] == 0


def test_complete_synthetic_manifest_requires_human_review_not_auto_count():
    outcome = audit_session(_synthetic_ready())
    assert outcome["status"] == "ready_for_human_evidence_review"
    assert outcome["unmet_gates"] == []
    assert outcome["new_independent_calibration_sessions_added"] == 0
    assert outcome["requires_original_evidence_and_human_review"] is True


def test_missing_rear_stereo_camera_blocks():
    record = _synthetic_ready()
    record["camera_setup"]["cameras"] = ["C1", "C6"]
    assert "required_camera_geometry_and_review" in audit_session(record)["unmet_gates"]


def test_holdout_overlap_or_nan_metric_blocks():
    record = _synthetic_ready()
    record["calibration"]["fit_holdout_disjoint_verified"] = False
    assert "frozen_disjoint_calibration_holdout" in audit_session(record)["unmet_gates"]
    record["calibration"]["fit_holdout_disjoint_verified"] = True
    record["calibration"]["heldout_board_quality_metric"]["value_px"] = float("nan")
    assert "frozen_disjoint_calibration_holdout" in audit_session(record)["unmet_gates"]


def test_reused_historical_session_name_not_accepted():
    record = _synthetic_ready()
    record["session_code"] = "S1"
    assert "recognized_new_session_metadata" in audit_session(record)["unmet_gates"]


def test_private_metadata_not_echoed_in_any_output():
    record = _synthetic_ready()
    record["camera_setup"]["site_code"] = "PRIVATE_SESSION_SITE_DO_NOT_PRINT"
    record["camera_setup"]["new_setup_or_remount_evidence"] = "PRIVATE_DOCUMENT_PATH_DO_NOT_PRINT"
    result = json.dumps(audit_session(record))
    assert "PRIVATE_SESSION_SITE" not in result
    assert "PRIVATE_DOCUMENT_PATH" not in result
    assert "CAL-SYNTHETIC-001" not in result


def test_cli_on_public_blank_template_exits_blocked_without_participant_details(capsys):
    status = main(["--session", str(TEMPLATE)])
    data = json.loads(capsys.readouterr().out)
    assert status == 2
    assert data["status"] == "blocked"
    assert data["new_independent_calibration_sessions_added"] == 0
