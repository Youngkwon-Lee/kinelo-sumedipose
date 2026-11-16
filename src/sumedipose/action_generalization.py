"""A3 squat versus A4 deadlift: same five calibration groups, descriptive only.

All participant-level inputs and RGB originals remain local. This module exports
only cross-group descriptive summaries, never per-subject or per-frame results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from .calibration_strata import (
    SESSIONS, CAMERAS, VARIANTS, MODALITIES,
    _get_calibration, spearman_descriptive, weighted_two_side_mae,
)

ACTIONS = ("A3", "A4")
MAX_FRAMES = 24


def _file_name(action: str, session: str, camera: str, variant: str) -> str:
    if action == "A3":
        return f"S1A3_pose_{session}_{camera}_{variant}.json"
    if action == "A4":
        return f"A4D2_pose_{session}_{camera}_{variant}.json"
    raise ValueError("Only A3 and A4 are supported")


def _validate(report: dict, action: str, session: str, camera: str,
              variant: str, expected_sha: str) -> None:
    expected_sample = f"{camera}{session}{action}D2"
    if report.get("sample") != expected_sample:
        raise ValueError(f"Wrong action/camera/participant in {_file_name(action,session,camera,variant)}")
    if report.get("model_variant") != variant or report.get("model_asset_sha256") != expected_sha:
        raise ValueError("MediaPipe model identity differs from locked A3 protocol")
    if (report.get("unique_images_sampled") != MAX_FRAMES
            or report.get("max_unique_images_requested") != MAX_FRAMES
            or report.get("inference_mode") != "IMAGE"):
        raise ValueError("Incompatible model-image sampling protocol")
    if report.get("model_detected_poses", 0) > MAX_FRAMES:
        raise ValueError("More poses than attempted images")


def analyse(calibration: dict, reports: dict, sha_by_variant: dict) -> dict:
    """Five groups, paired A3/A4; rho and medians are descriptive, not inferential."""
    group_residual = _get_calibration(calibration)
    missing = [
        _file_name(action, session, camera, variant)
        for action in ACTIONS for session in SESSIONS
        for camera in CAMERAS for variant in VARIANTS
        if (action, session, camera, variant) not in reports
    ]
    if missing:
        return {
            "status": "incomplete_missing_reports",
            "expected_model_reports": len(ACTIONS) * len(SESSIONS) * len(CAMERAS) * len(VARIANTS),
            "missing_model_reports": missing,
            "independent_calibration_groups": len(SESSIONS),
            "raw_images_or_subject_level_pose_metrics_exported": False,
        }
    for (action, session, camera, variant), report in reports.items():
        if action not in ACTIONS or session not in SESSIONS or camera not in CAMERAS or variant not in VARIANTS:
            raise ValueError("Unrecognized action/subject/camera/model key")
        _validate(report, action, session, camera, variant, sha_by_variant[variant])

    rows = []
    differences = []
    for camera in CAMERAS:
        for variant in VARIANTS:
            for modality in MODALITIES:
                by_action = {}
                for action in ACTIONS:
                    measures = [
                        (session, *weighted_two_side_mae(
                            reports[(action, session, camera, variant)], modality,
                            expected_frames=MAX_FRAMES))
                        for session in SESSIONS
                    ]
                    by_action[action] = {session: mae for session, mae, n in measures if n and mae is not None}
                    included = [(session, mae) for session, mae, n in measures if n and mae is not None]
                    ys = [mae for _, mae in included]
                    rows.append({
                        "action": action,
                        "camera": camera,
                        "model": variant,
                        "reference_space": modality,
                        "calibration_groups_with_valid_angles": len(included),
                        "spearman_rho_G65_vs_MAE_descriptive": spearman_descriptive(
                            [group_residual[session] for session, _ in included], ys),
                        "calibration_group_equal_weight_mae_median_deg": round(median(ys), 4) if ys else None,
                        "valid_side_angle_observations": sum(n for _, mae, n in measures if mae is not None),
                    })
                paired = [
                    (session, by_action["A4"][session] - by_action["A3"][session])
                    for session in SESSIONS
                    if session in by_action["A4"] and session in by_action["A3"]
                ]
                deltas = [delta for _, delta in paired]
                differences.append({
                    "camera": camera,
                    "model": variant,
                    "reference_space": modality,
                    "paired_calibration_groups": len(paired),
                    "A4_minus_A3_MAE_median_deg": round(median(deltas), 4) if deltas else None,
                    "spearman_rho_G65_vs_paired_MAE_difference_descriptive":
                        spearman_descriptive([group_residual[s] for s, _ in paired], deltas),
                    "group_level_difference_not_an_independent_calibration_trial": True,
                })
    complete = (all(row["calibration_groups_with_valid_angles"] == len(SESSIONS) for row in rows)
                and all(row["paired_calibration_groups"] == len(SESSIONS) for row in differences))
    return {
        "study": "A3 squat to A4 deadlift exploratory movement generalization",
        "status": "complete_exploratory" if complete else "incomplete_invalid_angles",
        "actions": list(ACTIONS),
        "repeat": "D2",
        "cameras": list(CAMERAS),
        "model_variants": list(VARIANTS),
        "unique_images_per_camera_model_subject": MAX_FRAMES,
        "expected_A4_model_image_inferences": len(SESSIONS)*len(CAMERAS)*len(VARIANTS)*MAX_FRAMES,
        "paired_participants": len(SESSIONS),
        "independent_calibration_groups": len(SESSIONS),
        "calibration_quality_proxy": "G65 (C6-C5) stereo Sampson median residual in pixels",
        "comparisons": rows,
        "within_participant_action_contrasts": differences,
        "interpretation_limits": [
            "Same five calibration groups in A3 and A4: no new independent calibrations",
            "n=5 group correlations are exploratory without significance testing",
            "G65 is a stereo residual, not directly measured monocular model accuracy",
            "2D projected-Vicon and 3D WCS angles have distinct reference geometries",
            "No independently proven exposure-time sync, joint-centre equivalence, or model-training disjointness",
            "Model world-landmark geometry is not validated clinical 3D knee kinematics",
        ],
        "clinical_accuracy_validated": False,
        "physical_timestamp_sync_verified": False,
        "raw_images_or_subject_level_pose_metrics_exported": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.analysis_dir
    calibration = json.loads((source/"S1_S6_S11_S17_S23_calibration_five_session_overview.json").read_text(encoding="utf-8"))
    protocol = json.loads((source/"S1A3_mediapipe_lite_full_protocol.json").read_text(encoding="utf-8"))
    expected_sha = {v: protocol["model_variants"][v]["sha256"] for v in VARIANTS}
    reports = {}
    for action in ACTIONS:
        for session in SESSIONS:
            for camera in CAMERAS:
                for variant in VARIANTS:
                    path = source/_file_name(action, session, camera, variant)
                    if path.exists():
                        reports[(action, session, camera, variant)] = json.loads(path.read_text(encoding="utf-8"))
    result = analyse(calibration, reports, expected_sha)
    repo = Path(__file__).resolve().parents[2]
    target = args.output.resolve()
    if target == repo or repo in target.parents:
        parser.error("No derived participant data or reports within public code checkout")
    if target.exists():
        parser.error("Output exists; refusing overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "independent_calibration_groups": result["independent_calibration_groups"],
        "expected_A4_model_image_inferences": result.get("expected_A4_model_image_inferences"),
    }))
    return 0 if result["status"] == "complete_exploratory" else 2


if __name__ == "__main__":
    raise SystemExit(main())
