"""A3/A4 D2 local-only index-shift and joint-angle bias pilot.

Tests WCS-record-index sensitivity, NOT camera/Vicon exposure synchronization.
MediaPipe landmarks and Vicon joint centres are anatomically different.
Images and participant-level diagnostics must remain on the research PC.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .benchmark import JOINT_NAMES, _angles_from_named, unique_sample_indices
from .geometry import joint_angle
from .io import read_sample

SHIFTS = (-12, -6, -3, 0, 3, 6, 12)
ANCHORS = ("S1", "S6", "S11", "S17", "S23")
CAMERAS = ("C1", "C6")
MODELS = ("lite", "full")
MODALITIES = ("image_2d", "world_3d")


def reference_angles(records, dimension: str) -> dict[str, np.ndarray]:
    """Calculate same-index projected or WCS Vicon knee included angles."""
    result = {}
    for side, names in JOINT_NAMES.items():
        values = []
        for record in records:
            try:
                values.append(_angles_from_named(record, names["reference"], dimension))
            except (ValueError, KeyError, IndexError, TypeError):
                values.append(float("nan"))
        result[side] = np.asarray(values, dtype=float)
    return result


def shift_sensitivity(
    pairs: list[tuple[int, str, float]],
    references: dict[str, np.ndarray],
    shifts: tuple[int, ...] = SHIFTS,
) -> dict:
    """Use exactly the same visible observations for all tested index shifts.

    'Best shift' is an in-sample diagnostic and MUST NOT be called a
    recovered time offset or validated temporal correction.
    """
    if 0 not in shifts or len(set(shifts)) != len(shifts):
        raise ValueError("shifts require unique values including zero")
    baseline = []
    matched = []
    for idx, side, prediction in pairs:
        if side not in references or not math.isfinite(float(prediction)):
            continue
        ref = references[side]
        if 0 <= idx < len(ref) and math.isfinite(float(ref[idx])):
            baseline.append(float(prediction) - float(ref[idx]))
        if all(0 <= idx + shift < len(ref) and math.isfinite(float(ref[idx + shift]))
               for shift in shifts):
            matched.append((float(prediction), [float(ref[idx + shift]) for shift in shifts]))
    if not baseline:
        return {"baseline_valid": 0, "matched_valid": 0,
                "baseline_mae_deg": None, "index_shift_is_time_sync_proof": False}
    base = np.asarray(baseline)
    result = {
        "baseline_valid": len(baseline),
        "baseline_mae_deg": round(float(np.mean(np.abs(base))), 4),
        "baseline_mean_signed_bias_deg": round(float(np.mean(base)), 4),
        "baseline_signed_error_sd_deg": round(float(np.std(base)), 4),
        "matched_valid": len(matched),
        "index_shift_is_time_sync_proof": False,
    }
    if not matched:
        return result
    p = np.asarray([row[0] for row in matched], dtype=float)
    refs = np.asarray([row[1] for row in matched], dtype=float)
    mae = np.mean(np.abs(p[:, None] - refs), axis=0)
    entries = {str(shift): round(float(value), 4) for shift, value in zip(shifts, mae)}
    best_i = int(np.argmin(mae))
    zero_i = shifts.index(0)
    result.update({
        "matched_zero_mae_deg": round(float(mae[zero_i]), 4),
        "shift_mae_degrees": entries,
        "best_exploratory_index_shift": shifts[best_i],
        "best_exploratory_mae_deg": round(float(mae[best_i]), 4),
        "best_vs_zero_in_sample_mae_delta_deg": round(float(mae[zero_i] - mae[best_i]), 4),
        "shift_zero_is_best_or_tied": bool(mae[zero_i] <= min(mae) + 1e-9),
        "same_observation_subset_at_every_shift": True,
    })
    return result


def evaluate_recording(root: Path, subject: str, camera: str, model_path: Path,
                       max_frames: int = 24, visibility: float = 0.5,
                       action: str = "A3") -> dict:
    """Run exactly the previous 24-image sampling rule; retain no landmarks."""
    if action not in ("A3", "A4"):
        raise ValueError("Only the A3 squat and A4 deadlift are supported")
    import cv2
    import mediapipe as mp

    paths, world, cam = read_sample(root, camera=camera, subject=subject, action=action, repeat="D2")
    if len(world) != len(cam):
        raise ValueError("WCS/camera coordinate record count mismatch")
    refs = {
        "image_2d": reference_angles(cam, "xy"),
        "world_3d": reference_angles(world, "xyz"),
    }
    file_counts = Counter(Path(str(record.get("path", ""))).name for record in cam)
    predictions = {modality: [] for modality in MODALITIES}
    detections = 0
    with ZipFile(paths["rgb"]) as archive:
        picks, missing = unique_sample_indices(cam, archive.namelist(), max_frames)
        if len(picks) != max_frames:
            raise ValueError("Cannot compare against frozen 24-unique-frame protocol")
        selected_with_reused_record = sum(file_counts.get(Path(name).name, 0) > 1
                                          for name, _ in picks)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1)
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as task:
            for name, idx in picks:
                jpg = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if jpg is None:
                    continue
                h, w = jpg.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB))
                detection = task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not detection.pose_landmarks or not detection.pose_world_landmarks:
                    continue
                detections += 1
                image_pts = detection.pose_landmarks[0]
                world_pts = detection.pose_world_landmarks[0]
                for side, spec in JOINT_NAMES.items():
                    positions = spec["pred"]
                    if any(image_pts[j].visibility < visibility
                           or world_pts[j].visibility < visibility for j in positions):
                        continue
                    pred2 = [[image_pts[j].x * w, image_pts[j].y * h] for j in positions]
                    pred3 = [[world_pts[j].x, world_pts[j].y, world_pts[j].z]
                             for j in positions]
                    for modality, coordinates in (("image_2d", pred2), ("world_3d", pred3)):
                        angle = joint_angle(*coordinates)
                        if math.isfinite(angle):
                            predictions[modality].append((idx, side, angle))
    return {
        "subject": subject, "camera": camera, "action": action, "model": model_path.stem,
        "unique_rgb_images_sampled": len(picks),
        "poses_detected": detections,
        "wcs_coordinate_records": len(world),
        "camera_coordinate_records": len(cam),
        "distinct_rgb_files_referenced": len(file_counts),
        "rgb_reused_coordinate_records": sum(n - 1 for n in file_counts.values()),
        "selected_rgb_with_multiple_coordinate_records": selected_with_reused_record,
        "source_image_references_unavailable": missing,
        "modalities": {
            modality: shift_sensitivity(predictions[modality], refs[modality])
            for modality in MODALITIES
        },
        "actual_exposure_time_synchronization_verified": False,
        "landmark_vs_joint_centre_definition_aligned": False,
    }


def aggregate_pilot(rows: list[dict], action: str = "A3") -> dict:
    """Aggregate across five anchors without leaking participant identifiers."""
    if action not in ("A3", "A4") or any(row.get("action", action) != action for row in rows):
        raise ValueError("Mixed or unsupported action in frame-index pilot")
    table = []
    for camera in CAMERAS:
        for model in MODELS:
            for modality in MODALITIES:
                selected = [
                    row["modalities"][modality]
                    for row in rows if row["camera"] == camera and row["model"].endswith(model)
                ]
                if len(selected) != len(ANCHORS):
                    raise ValueError("Expected five anchor diagnostics per view/model/modality")
                available = [r for r in selected if r.get("baseline_mae_deg") is not None
                             and r.get("matched_zero_mae_deg") is not None]
                if not available:
                    raise ValueError("No valid angular pairs in a pilot cell")
                table.append({
                    "camera": camera, "model": model, "reference_space": modality,
                    "anchor_sessions": len(selected),
                    "median_baseline_mae_degrees": round(float(np.median(
                        [r["baseline_mae_deg"] for r in available])), 4),
                    "median_absolute_signed_bias_degrees": round(float(np.median(
                        [abs(r["baseline_mean_signed_bias_deg"]) for r in available])), 4),
                    "median_in_sample_best_shift_improvement_deg": round(float(np.median(
                        [r["best_vs_zero_in_sample_mae_delta_deg"] for r in available])), 4),
                    "anchors_zero_shift_best_or_tied": sum(
                        bool(r["shift_zero_is_best_or_tied"]) for r in available),
                    "matched_valid_angle_samples": sum(r["matched_valid"] for r in available),
                })
    return {
        "status": "exploratory_index_shift_and_signed_bias_pilot",
        "action": "A3 squat" if action == "A3" else "A4 deadlift", "repeat": "D2",
        "n_anchor_recordings": len(ANCHORS),
        "independent_calibration_groups": len(ANCHORS),
        "new_independent_calibration_groups": 0,
        "camera_views": list(CAMERAS), "models": list(MODELS),
        "tested_WCS_record_index_shifts": list(SHIFTS),
        "inference_reports": len(rows),
        "image_inferences": sum(r["unique_rgb_images_sampled"] for r in rows),
        "pose_detections": sum(r["poses_detected"] for r in rows),
        "aggregate_cells": table,
        "safety": {
            "in_sample_shift_selection_not_hardware_time_sync": True,
            "signed_angle_bias_not_proven_joint_centre_cause": True,
            "Vicon_MediaPipe_joint_definitions_differ": True,
            "2D_3D_MAE_not_directly_comparable": True,
            "S1_used_in_prior_ROI_tuning_not_independent_holdout": True,
            "clinical_accuracy_validated": False,
            "raw_images_or_per_frame_landmarks_saved": False,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--action", choices=("A3", "A4"), default="A3",
                        help="A3 squat (default) or A4 deadlift; D2 only")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    outputs = [args.output.resolve(), args.aggregate_output.resolve()]
    if len(set(outputs)) != len(outputs):
        parser.error("Detailed and aggregate destinations must differ")
    for path in outputs:
        if path.exists():
            parser.error(f"Refuse overwrite: {path}")
        if path == repository or repository in path.parents:
            parser.error("Participant-derived output forbidden inside code repository")
        if args.analysis_dir.resolve() != path.parent:
            parser.error("Save results only in explicitly selected analysis directory")
    protocol = json.loads((args.analysis_dir /
        "A3_new_subject_generalization_protocol_20261116.json").read_text(encoding="utf8"))
    model_paths = {}
    for model in MODELS:
        path = args.model_dir / f"pose_landmarker_{model}.task"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != protocol["models"][model]["sha256"]:
            parser.error(f"Frozen model asset SHA mismatch: {model}")
        model_paths[model] = path
    reports = []
    for subject in ANCHORS:
        for camera in CAMERAS:
            for model in MODELS:
                reports.append(evaluate_recording(
                    args.root, subject, camera, model_paths[model], action=args.action))
    aggregate = aggregate_pilot(reports, action=args.action)
    local = {
        "status": aggregate["status"],
        "action": args.action,
        "subjects": list(ANCHORS),
        "model_asset_sha256": {m: protocol["models"][m]["sha256"] for m in MODELS},
        "reports": reports,
        "source_data_remains_local": True,
        "no_raw_images_or_per_frame_keypoints_saved": True,
    }
    for path, payload in ((args.output, local), (args.aggregate_output, aggregate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps({k: aggregate[k] for k in (
        "status", "n_anchor_recordings", "inference_reports",
        "image_inferences", "pose_detections", "aggregate_cells")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
