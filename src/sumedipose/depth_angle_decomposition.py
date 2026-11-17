"""Local-only A3/D2 camera-plane vs 3D knee-angle discrepancy decomposition.

Not a decomposition of causal error sources. Source camera XYZ and MediaPipe
world XYZ have different anatomical landmark definitions and coordinate frames.
Only included angles and their planar/depth *terms* are compared; the model's
landmark coordinates are never aligned or interpreted as physical Vicon points.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import numpy as np

from .benchmark import JOINT_NAMES, unique_sample_indices
from .position_holdout import strict_unique_pairs
from .calibration_strata import weighted_two_side_mae
from .geometry import joint_angle
from .io import read_sample

ANCHORS = ("S1", "S6", "S11", "S17", "S23")
GROUPS = {s: ("S1" if i <= 5 else "S6" if i <= 10 else
              "S11" if i <= 16 else "S17" if i <= 22 else "S23")
          for i in range(1, 29) for s in (f"S{i}",) if i not in (5, 10)}
CAMERAS = ("C1", "C6")
MODELS = ("lite", "full")
FIELDS = (
    "signed_3d_error_cam", "signed_3d_error_wcs",
    "signed_flat_xy_error", "signed_z_component_gap",
    "signed_image2d_error", "reference_z_angle_effect",
    "model_z_angle_effect", "reference_cam_wcs_angle_gap",
    "abs_3d_error_cam", "abs_3d_error_wcs", "abs_flat_xy_error",
    "abs_z_component_gap", "abs_image2d_error",
    "abs_segment_ratio_log_gap", "abs_reference_cam_wcs_angle_gap",
    "abs_reference_2d_projection_gap", "abs_model_flat_vs_image2d_gap",
)
PHASES = ("flexed_under_120", "middle_120_to_150", "extended_150_or_more")


def phase_bin(reference_wcs_included_angle: float) -> str:
    if reference_wcs_included_angle < 120:
        return PHASES[0]
    if reference_wcs_included_angle < 150:
        return PHASES[1]
    return PHASES[2]


def _three(record: dict, names: tuple[str, ...], key: str) -> np.ndarray:
    ids = record["point_ids"]
    return np.asarray([record[key][ids.index(s)] for s in names], dtype=float)


def _segment_ratio(points: np.ndarray) -> float:
    thigh = float(np.linalg.norm(points[0] - points[1]))
    shank = float(np.linalg.norm(points[2] - points[1]))
    return thigh / shank if thigh > 1e-9 and shank > 1e-9 else float("nan")


def angle_terms(pred3: np.ndarray, refcam3: np.ndarray, refwcs3: np.ndarray,
                pred2: np.ndarray, ref2: np.ndarray) -> dict:
    """Exact angle identity: e3D_cam = eXY_flat + (depth term mismatch).

    This is a *geometry of two model/source representations*, not proof of
    physical Z-coordinate errors, joint-centre anatomy or time alignment.
    """
    p3, r3 = joint_angle(*pred3), joint_angle(*refcam3)
    w3 = joint_angle(*refwcs3)
    pxy, rxy = joint_angle(*(pred3[:, :2])), joint_angle(*(refcam3[:, :2]))
    p2, r2 = joint_angle(*pred2), joint_angle(*ref2)
    vals = [p3, r3, w3, pxy, rxy, p2, r2]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Non-finite joint angle")
    if abs(r3 - w3) > 1:
        raise ValueError("Vicon camera 3D included angle deviates >1 degree from WCS")
    plane = pxy - rxy
    z_term = (p3 - pxy) - (r3 - rxy)
    total = p3 - r3
    if abs(total - (plane + z_term)) > 1e-8:
        raise AssertionError("Additive geometry identity failed")
    ratios = (_segment_ratio(pred3), _segment_ratio(refcam3))
    if not all(math.isfinite(x) and x > 0 for x in ratios):
        raise ValueError("Degenerate 3D limb length ratio")
    log_ratio_gap = math.log(ratios[0] / ratios[1])
    return {
        "signed_3d_error_cam": total,
        "signed_3d_error_wcs": p3 - w3,
        "signed_flat_xy_error": plane,
        "signed_z_component_gap": z_term,
        "signed_image2d_error": p2 - r2,
        "reference_z_angle_effect": r3 - rxy,
        "model_z_angle_effect": p3 - pxy,
        "reference_cam_wcs_angle_gap": r3 - w3,
        "abs_3d_error_cam": abs(total),
        "abs_3d_error_wcs": abs(p3 - w3),
        "abs_flat_xy_error": abs(plane),
        "abs_z_component_gap": abs(z_term),
        "abs_image2d_error": abs(p2 - r2),
        "abs_segment_ratio_log_gap": abs(log_ratio_gap),
        "abs_reference_cam_wcs_angle_gap": abs(r3 - w3),
        "abs_reference_2d_projection_gap": abs(rxy - r2),
        "abs_model_flat_vs_image2d_gap": abs(pxy - p2),
        "phase": phase_bin(w3),
    }


def _mean_metrics(observations: list[dict]) -> dict:
    return {"valid_joint_angles": len(observations),
            **{field: round(mean(o[field] for o in observations), 6) if observations else None
               for field in FIELDS}}


def _strict_report(analysis: Path, subject: str, camera: str, model: str) -> dict:
    """Existing source-valid paired rerun: no new inference to establish reference."""
    path = (analysis / "strict_source_pairing_20261117" /
            f"strict_A3D2_{subject}_{camera}_{model}.json")
    if not path.is_file():
        raise ValueError("Missing previously verified source-valid D2 report")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if (report.get("pairing", {}).get("new_reference_uses_dup_num_equals_factor")
            is not True):
        raise ValueError("Previous reference did not use source-valid pairing")
    return report


def evaluate_one(root: Path, analysis: Path, model_path: Path, subject: str,
                 camera: str, model: str) -> dict:
    import cv2
    import mediapipe as mp

    paths, world, cam = read_sample(root, camera=camera, subject=subject,
                                    action="A3", repeat="D2")
    if len(world) != len(cam) or not world:
        raise ValueError("WCS/camera index alignment not present")
    observed = []
    observed_by_side = {side: [] for side in JOINT_NAMES}
    detected = 0
    with ZipFile(paths["rgb"]) as archive:
        old, _ = unique_sample_indices(cam, archive.namelist(), max_frames=24)
        picks, pairing_qa = strict_unique_pairs(cam, archive.namelist(), count=24)
        if (len(old) != 24 or len(picks) != 24 or
                [name for name, _ in old] != [name for name, _ in picks] or
                pairing_qa["selected_jpg_names_different_from_previous"] != 0 or
                pairing_qa["ambiguous_or_unverifiable_jpg_groups"] != 0):
            raise ValueError("Frozen RGB set changed or strict association is ambiguous")
        if any(cam[idx].get("dup_num") != cam[idx].get("factor")
               for _, idx in picks):
            raise ValueError("Source-valid coordinate associations not enforced")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1)
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as task:
            for member, idx in picks:
                jpg = cv2.imdecode(np.frombuffer(archive.read(member), np.uint8),
                                   cv2.IMREAD_COLOR)
                if jpg is None:
                    raise ValueError("Unreadable original RGB sample")
                h, w = jpg.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB))
                result = task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.pose_landmarks or not result.pose_world_landmarks:
                    continue
                detected += 1
                im, xyz = result.pose_landmarks[0], result.pose_world_landmarks[0]
                if world[idx]["point_ids"] != cam[idx]["point_ids"]:
                    raise ValueError("WCS and camera marker identity mismatch")
                for side, names in JOINT_NAMES.items():
                    inds = names["pred"]
                    if any(im[j].visibility < .5 or xyz[j].visibility < .5 for j in inds):
                        continue
                    pred3 = np.asarray([[xyz[j].x, xyz[j].y, xyz[j].z] for j in inds])
                    pred2 = np.asarray([[im[j].x * w, im[j].y * h] for j in inds])
                    try:
                        obs = angle_terms(
                            pred3, _three(cam[idx], names["reference"], "xyz"),
                            _three(world[idx], names["reference"], "xyz"),
                            pred2, _three(cam[idx], names["reference"], "xy"),
                        )
                    except (ValueError, IndexError, KeyError, TypeError):
                        continue
                    observed.append(obs)
                    observed_by_side[side].append(obs)
    if not observed:
        raise ValueError("No valid side-angle observations")
    if detected != 24:
        raise ValueError("Prior census found 24/24 poses: unexpected change")
    prior = _strict_report(analysis, subject, camera, model)
    if (prior["sample"] != f"{camera}{subject}A3D2"
            or prior["model_variant"] != model
            or prior["model_asset_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest()
            or prior["model_detected_poses"] != detected):
        raise ValueError("Previous strict report/model asset/detection mismatch")
    current = _mean_metrics(observed)
    for modality, field in (("image_2d", "abs_image2d_error"),
                            ("world_3d", "abs_3d_error_wcs")):
        old_mae, old_count = weighted_two_side_mae(
            {"results": prior["strict_results"]}, modality, expected_frames=24)
        if (old_count != len(observed)
                or abs(old_mae - mean(o[field] for o in observed)) > 0.005):
            raise ValueError(f"Source-valid total MAE or count mismatch: {modality}")
        for side in JOINT_NAMES:
            baseline = prior["strict_results"][modality][side]
            measured = observed_by_side[side]
            if (baseline["valid"] != len(measured) or
                    not measured or
                    abs(baseline["MAE_degrees"] -
                        mean(o[field] for o in measured)) > 0.005):
                raise ValueError(f"Source-valid {side}/{modality} baseline mismatch")
    return {
        "subject": subject, "calibration_group": GROUPS[subject],
        "camera": camera, "model": model, "unique_images": 24, "pose_detections": detected,
        "all": current,
        "by_phase": {phase: _mean_metrics([o for o in observed if o["phase"] == phase])
                     for phase in PHASES},
        "prior_source_valid_D2_MAE_and_valid_count_readback": "passed",
        "source_valid_pairing": True,
        "changed_original_reference_indices": pairing_qa["previous_index_selection_invalid_pairs"],
        "source_missing_original_JPG_groups": pairing_qa["missing_original_jpg_groups"],
    }


def _group_equal_summary(rows: list[dict], camera: str, model: str, phase: str) -> dict:
    selected = [x for x in rows if x["camera"] == camera and x["model"] == model]
    grouped = defaultdict(list)
    for row in selected:
        d = row["all"] if phase == "all" else row["by_phase"][phase]
        if d["valid_joint_angles"]:
            grouped[row["calibration_group"]].append(d)
    if not grouped:
        return {"phase": phase, "groups_with_data": 0, "valid_joint_angles": 0}
    result = {"camera": camera, "model": model, "phase": phase,
              "groups_with_data": len(grouped),
              "valid_joint_angles": sum(d["valid_joint_angles"] for items in grouped.values()
                                        for d in items),
              "participants_with_valid_angles": sum(len(items) for items in grouped.values())}
    for field in FIELDS:
        values = [mean(d[field] for d in items) for items in grouped.values()]
        result[field + "_mean_of_group_means"] = round(mean(values), 4)
        result[field + "_median_group_mean"] = round(median(values), 4)
    return result


def grouped_summary(rows: list[dict], scope: str) -> dict:
    subjects = {r["subject"] for r in rows}
    expected = set(ANCHORS if scope == "anchors" else GROUPS)
    if subjects != expected or len(rows) != len(subjects) * 4:
        raise ValueError("Incomplete cohort or camera/model crossing")
    if scope == "available26" and {
        key:sum(1 for s in subjects if GROUPS[s] == key) for key in ANCHORS
    } != dict(zip(ANCHORS, (4, 4, 6, 6, 6))):
        raise ValueError("Calibration-group census sizes unexpectedly changed")
    return {
        "status": "complete_exploratory_geometry_decomposition",
        "action": "A3", "repeat": "D2", "scope": scope,
        "n_subjects": len(subjects), "n_calibration_groups": 5,
        "new_independent_calibration_groups": 0,
        "model_reports": len(rows),
        "model_image_inferences": sum(r["unique_images"] for r in rows),
        "model_pose_detections": sum(r["pose_detections"] for r in rows),
        "all_prior_source_valid_report_MAE_and_valid_count_checks_passed": True,
        "source_valid_pairs_dup_num_equals_factor": True,
        "original_image_filenames_preserved": True,
        "formula": "angle3D_model-angle3D_ViconCamera = (angleXY_model-angleXY_ViconCamera) + ((angle3D_model-angleXY_model)-(angle3D_ViconCamera-angleXY_ViconCamera))",
        "phase_bins": {"flexed_under_120": "[0,120)", "middle_120_to_150": "[120,150)",
                       "extended_150_or_more": "[150,180]"},
        "results": [_group_equal_summary(rows, cam, model, phase)
                    for cam in CAMERAS for model in MODELS
                    for phase in ("all", *PHASES)],
        "interpretation": {
            "camera_xyz_3d_is_within_one_degree_of_wcs_in_all_used_samples": True,
            "model_world_and_vicon_camera_xy_axes_not_independently_registered": True,
            "out_of_plane_component_not_proven_camera_depth_error": True,
            "strict_source_valid_reference_used": True,
            "angle_plane_decomposition_is_algebra_not_causal_depth_error": True,
            "world_landmarks_and_vicon_joint_centres_not_anatomically_equivalent": True,
            "flattened_model_xy_and_camera_xy_not_identical_calibrated_physical_planes": True,
            "image2d_and_world3d_reference_spaces_separate": True,
            "oracle_depth_replacement_not_a_deployable_calibration": True,
            "independent_camera_vicon_capture_time_not_verified": True,
            "clinical_accuracy_validated": False,
            "no_raw_images_or_frame_or_landmarks_saved": True,
            "no_inferential_claim_or_new_calibration_session": True,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["anchors", "available26"], required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    outputs = (args.private_output.resolve(), args.aggregate_output.resolve())
    if len(set(outputs)) != 2:
        parser.error("Output destinations must be distinct")
    for target in outputs:
        if target.exists():
            parser.error(f"Refuse to overwrite existing report: {target}")
        if target.parent != args.analysis_dir.resolve() or target == repo or repo in target.parents:
            parser.error("Only explicit private Windows analysis outputs allowed")
    protocol = json.loads((args.analysis_dir / "A3_new_subject_generalization_protocol_20261116.json").read_text(encoding="utf8"))
    assets = {}
    for model in MODELS:
        path = args.model_dir / f"pose_landmarker_{model}.task"
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["models"][model]["sha256"]:
            parser.error(f"Original MediaPipe model SHA256 changed: {model}")
        assets[model] = path
    subjects = ANCHORS if args.scope == "anchors" else tuple(GROUPS)
    rows = []
    for subject in subjects:
        for camera in CAMERAS:
            for model in MODELS:
                rows.append(evaluate_one(args.root, args.analysis_dir,
                                         assets[model], subject, camera, model))
    report = grouped_summary(rows, args.scope)
    private = {"status": report["status"], "scope": args.scope,
               "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "sampled_participant_summaries": rows,
               "raw_RGB_keypoints_or_frame_level_values_saved": False}
    for path, payload in ((args.private_output, private),
                          (args.aggregate_output, report)):
        with path.open("x", encoding="utf8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
    print(json.dumps({
        "status": report["status"], "scope": args.scope,
        "n_subjects": report["n_subjects"],
        "model_reports": report["model_reports"],
        "image_inferences": report["model_image_inferences"],
        "detected": report["model_pose_detections"],
        "comparisons": [
            {k: r[k] for k in ("camera", "model", "phase",
              "valid_joint_angles", "signed_3d_error_cam_mean_of_group_means",
              "signed_flat_xy_error_mean_of_group_means",
              "signed_z_component_gap_mean_of_group_means",
              "abs_3d_error_cam_mean_of_group_means",
              "abs_flat_xy_error_mean_of_group_means",
              "abs_z_component_gap_mean_of_group_means")}
            for r in report["results"] if r["phase"] == "all"
        ],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
