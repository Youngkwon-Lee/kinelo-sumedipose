"""Fail-closed, non-identifying readiness gate for a future calibration session.

A filled checklist is *not* independent verification of its claims. This tool
never increments the number of independent calibration sessions and never
prints input metadata, subject information, source paths, or hardware evidence.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import re


def _dig(record: dict, *keys: str):
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _true(record: dict, *path: str) -> bool:
    return _dig(record, *path) is True


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _valid_date(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _verified_reference(value) -> bool:
    return isinstance(value, str) and value not in {
        "", "UNKNOWN", "UNVERIFIED", "PLACEHOLDER", "NOT_COMPUTED",
        "VERSION_PLACEHOLDER", "CODE_ONLY",
    } and "PLACEHOLDER" not in value.upper()


def _valid_metric(value) -> bool:
    return (
        isinstance(value, dict)
        and value.get("name") == "C6_C5_Sampson_median_px"
        and type(value.get("value_px")) in (int, float)
        and math.isfinite(value["value_px"])
        and value["value_px"] >= 0
    )


def audit_session(record: dict) -> dict:
    """Inspect self-reported evidence; never grant independent-session status."""
    if not isinstance(record, dict):
        record = {}
    session = _dig(record, "session_code")
    historical = {"S1", "S6", "S11", "S17", "S23"}
    session_tag_ok = (
        isinstance(session, str)
        and re.fullmatch(r"CAL-[A-Z0-9][A-Z0-9_-]{3,}", session) is not None
        and "PLACEHOLDER" not in session
        and session not in historical
    )
    views = _dig(record, "camera_setup", "cameras")
    actions = _dig(record, "movement_plan", "actions")
    pose_views = _dig(record, "movement_plan", "views_for_pose")
    model_set = _dig(record, "movement_plan", "models")
    scale = _dig(record, "calibration", "square_size_mm")
    fit_count = _dig(record, "calibration", "C6_C5_fit_frames_count")
    holdout_count = _dig(record, "calibration", "C6_C5_heldout_frames_count")
    equipment = _dig(record, "equipment") or {}
    equipment_checks = (
        "front_C1_reserved", "rear_C6_reserved", "stereo_C5_reserved",
        "Vicon_reserved", "metric_checkerboard_confirmed",
    )
    gates = {
        "recognized_new_session_metadata": (
            _dig(record, "protocol") == "independent-calibration-study-v1"
            and _dig(record, "research_status") == "collected_for_review"
            and session_tag_ok
            and _valid_date(_dig(record, "acquisition_date"))
        ),
        "equipment_availability_confirmed": all(
            equipment.get(key) is True for key in equipment_checks
        ),
        "new_calibration_capture_evidence": (
            _true(record, "camera_setup", "new_calibration_acquisition")
            and _verified_reference(_dig(record, "camera_setup", "site_code"))
            and _verified_reference(_dig(record, "camera_setup", "rig_config_version"))
            and _verified_reference(
                _dig(record, "camera_setup", "new_setup_or_remount_evidence")
            )
        ),
        "required_camera_geometry_and_review": (
            isinstance(views, list)
            and {"C1", "C5", "C6"}.issubset(set(str(v) for v in views))
            and _dig(record, "camera_setup",
                     "metric_compatible_with_SUMediPose_G65") == "VERIFIED"
        ),
        "frozen_disjoint_calibration_holdout": (
            type(scale) in (int, float) and math.isfinite(scale)
            and abs(scale - 60.0) < 0.001
            and _positive_int(fit_count)
            and _positive_int(holdout_count)
            and _true(record, "calibration", "fit_holdout_disjoint_verified")
            and _true(record, "calibration", "intrinsics_frozen_before_holdout")
            and _true(record, "calibration", "extrinsics_frozen_before_holdout")
            and _valid_metric(_dig(record, "calibration", "heldout_board_quality_metric"))
        ),
        "physical_time_reference_reviewed": (
            _true(record, "sync", "camera_exposure_timestamps_observed")
            and _true(record, "sync", "Vicon_reference_clock_verified")
            and _true(record, "sync", "common_trigger_or_independent_event_verified")
            and _dig(record, "sync", "physical_timestamp_alignment_status")
                == "EVIDENCE_REVIEWED"
        ),
        "Vicon_and_joint_reference_reviewed": (
            _true(record, "reference", "camera_Vicon_registration_verified")
            and _true(record, "reference", "anatomical_joint_definition_reviewed")
        ),
        "locked_action_and_model_sampling": (
            isinstance(actions, list) and {"A3", "A4"}.issubset(actions)
            and isinstance(pose_views, list)
            and {"C1", "C6"}.issubset(pose_views)
            and isinstance(model_set, list)
            and {"lite", "full"}.issubset(model_set)
            and _dig(record, "movement_plan", "unique_rgb_frames_per_recording") == 24
            and _true(record, "movement_plan", "participant_records_private")
        ),
        "permissions_confirmed": all(
            _true(record, "permissions", field)
            for field in (
                "research_permission_verified",
                "participant_consent_verified",
                "usage_or_license_verified",
            )
        ),
        "no_premature_independence_claim": (
            _dig(record, "outcome", "count_as_new_independent_unit") is False
        ),
    }
    blocked = sorted(key for key, passed in gates.items() if not passed)
    return {
        "status": "blocked" if blocked else "ready_for_human_evidence_review",
        "unmet_gates": blocked,
        "checked_gate_count": len(gates),
        "new_independent_calibration_sessions_added": 0,
        "requires_original_evidence_and_human_review": True,
        "input_metadata_or_participant_records_exported": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path,
                        help="Locally stored private session intake, never a public Git file")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.session.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        print(json.dumps({"status": "invalid_local_input",
                          "new_independent_calibration_sessions_added": 0}))
        return 2
    outcome = audit_session(record)
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome["status"] == "ready_for_human_evidence_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
