"""Exploratory held-out root-relative 3D geometry, NOT physical MPJPE.

A single similarity transform is fitted on the first half of A3 RGB frames
for each participant/view/model and frozen before evaluating withheld A3
and all A4. The coordinates are hip-root-centred for *each* frame, so absolute
root positions, camera/Vicon registration, joint-anatomy equivalence and
physical timestamps are NOT evaluated.

Source RGB, Vicon coordinates, per-subject results and transforms remain local.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .benchmark import unique_sample_indices
from .io import read_sample

ANCHORS = ("S1", "S6", "S11", "S17", "S23")
CAMERAS = ("C1", "C6")
VARIANTS = ("lite", "full")
ACTIONS = ("A3", "A4")
GT_JOINTS = ("LHJC", "LKJC", "LAJC", "RHJC", "RKJC", "RAJC")
MODEL_JOINTS = (23, 25, 27, 24, 26, 28)
SAMPLED = 24
TRAIN_SLOTS = 12
MIN_VALID_PER_SET = 6


def strict_unique_pairs(records, image_members, count=SAMPLED):
    """Only the source's true frame (dup_num==factor) can label an RGB JPG.

    Dropped frames can reuse the NEXT image path as a placeholder. Previous
    mid-index selection could accidentally choose a missing/dropped record.
    """
    if type(count) is not int or count < 2:
        raise ValueError("count must be >=2")
    members = set(image_members)
    groups = OrderedDict()
    for i, record in enumerate(records):
        name = Path(str(record.get("path", ""))).name
        if name:
            groups.setdefault(name, []).append(i)
    valid = []
    ambiguous = 0
    missing = 0
    for name, indices in groups.items():
        if name not in members:
            missing += 1
            continue
        correct = [
            i for i in indices
            if type(records[i].get("factor")) is int
            and records[i]["factor"] > 0
            and type(records[i].get("dup_num")) is int
            and records[i]["dup_num"] == records[i]["factor"]
        ]
        if len(correct) == 1:
            valid.append((name, correct[0]))
        else:
            ambiguous += 1
    if len(valid) < count:
        raise ValueError("Insufficient uniquely timestamp-mapped JPG references")
    picks = [valid[int(i)] for i in np.linspace(0, len(valid) - 1, count, dtype=int)]
    if len(set(n for n, _ in picks)) != count:
        raise ValueError("Nonunique RGB sample")
    old, _ = unique_sample_indices(records, image_members, count)
    old_mispaired = sum(
        records[i].get("dup_num") != records[i].get("factor")
        for _, i in old
    )
    old_names = set(n for n, _ in old)
    return picks, {
        "available_source_path_groups": len(groups),
        "missing_original_jpg_groups": missing,
        "ambiguous_or_unverifiable_jpg_groups": ambiguous,
        "previous_index_selection_invalid_pairs": old_mispaired,
        "selected_jpg_names_different_from_previous": len(
            old_names.symmetric_difference({n for n, _ in picks})
        ) // 2,
        "unique_verified_jpgs_selected": len(picks),
    }


def root_relative(points):
    """Same hip-centre subtraction separately for model and Vicon."""
    xyz = np.asarray(points, dtype=float)
    if xyz.shape != (6, 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected six finite 3D joint positions")
    return xyz - (xyz[0] + xyz[3]) / 2.0


def fit_fixed_similarity(predicted: np.ndarray, reference: np.ndarray):
    """Fit a single proper rotation and positive scale, no per-frame fitting."""
    x, y = np.asarray(predicted, float), np.asarray(reference, float)
    if x.ndim != 3 or x.shape[1:] != (6, 3) or y.shape != x.shape or len(x) < MIN_VALID_PER_SET:
        raise ValueError("Need >=6 paired training frames, shape [N,6,3]")
    a, b = x.reshape(-1, 3), y.reshape(-1, 3)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.sum(a*a) < 1e-8:
        raise ValueError("Degenerate or invalid training coordinates")
    u, singular, vt = np.linalg.svd(a.T @ b)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = float(np.sum((a @ rotation) * b) / np.sum(a*a))
    if not np.isfinite(scale) or scale <= 0 or np.linalg.det(rotation) < 0.999:
        raise ValueError("Invalid or reflected similarity alignment")
    return rotation, scale


def heldout_discrepancy(frames, rotation, scale):
    """Return only aggregate position differences, normalized to GT femur length.

    This is *not* joint-location MPJPE in real-world XYZ. Each frame's GT root
    is used for root centering, and model->Vicon scale/orientation is fitted.
    """
    if len(frames) < MIN_VALID_PER_SET:
        raise ValueError("Insufficient held-out valid six-joint frames")
    per_frame_raw, per_frame_normalized = [], []
    for _, model, gt in frames:
        delta = np.linalg.norm(scale * (model @ rotation) - gt, axis=1)
        femur = np.mean([np.linalg.norm(gt[0] - gt[1]),
                         np.linalg.norm(gt[3] - gt[4])])
        if not np.isfinite(femur) or femur < 1e-8:
            raise ValueError("Degenerate reference femur segment")
        per_frame_raw.append(float(delta.mean()))
        per_frame_normalized.append(float(100.0 * delta.mean() / femur))
    return {
        "valid_evaluation_frames": len(frames),
        "mean_6joint_discrepancy_WCS_native_units": round(
            float(np.mean(per_frame_raw)), 4),
        "mean_discrepancy_pct_reference_femur_length": round(
            float(np.mean(per_frame_normalized)), 4),
        "median_discrepancy_pct_reference_femur_length": round(
            float(np.median(per_frame_normalized)), 4),
        "absolute_root_translation_measured": False,
        "physical_mpjpe_validated": False,
    }


def read_pose_frames(root, model_asset, action, camera, subject, *,
                     count=SAMPLED, visibility=0.5):
    """Infer in-memory only. Never return/save RGB or per-frame keypoints."""
    import cv2
    import mediapipe as mp

    paths, wcs, camera_records = read_sample(
        root, camera=camera, subject=subject, action=action, repeat="D2")
    if len(wcs) != len(camera_records):
        raise ValueError("WCS and camera source indices differ")
    with ZipFile(paths["rgb"]) as archive:
        picks, quality = strict_unique_pairs(
            camera_records, archive.namelist(), count)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_asset)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1)
        frames = []
        detected = 0
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for slot, (name, index) in enumerate(picks):
                image = cv2.imdecode(
                    np.frombuffer(archive.read(name), np.uint8),
                    cv2.IMREAD_COLOR)
                if image is None:
                    continue
                rgb = np.ascontiguousarray(cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB))
                result = detector.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.pose_world_landmarks or not result.pose_landmarks:
                    continue
                detected += 1
                lm = result.pose_world_landmarks[0]
                xy = result.pose_landmarks[0]
                if any(lm[i].visibility < visibility or
                       xy[i].visibility < visibility for i in MODEL_JOINTS):
                    continue
                try:
                    wcs_record = wcs[index]
                    cam_record = camera_records[index]
                    if wcs_record["point_ids"] != cam_record["point_ids"]:
                        raise ValueError("Vicon point IDs do not match")
                    gt_ids = [wcs_record["point_ids"].index(n) for n in GT_JOINTS]
                    gt = root_relative(
                        np.asarray(wcs_record["xyz"], float)[gt_ids])
                    model = root_relative(np.asarray([
                        [lm[i].x, lm[i].y, lm[i].z]
                        for i in MODEL_JOINTS
                    ], float))
                    femur = min(np.linalg.norm(gt[0] - gt[1]),
                                np.linalg.norm(gt[3] - gt[4]))
                    if femur < 1e-8 or not np.isfinite(femur):
                        continue
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                frames.append((slot, model, gt))
    quality.update({
        "model_images_attempted": len(picks),
        "pose_detections": detected,
        "valid_six_joint_frames": len(frames),
    })
    return frames, quality


def run_one(root, subject, camera, model, path):
    """Train on A3 first 12, test on A3 last 12 and A4 all 24."""
    squat, a3_source = read_pose_frames(
        root, path, "A3", camera, subject)
    deadlift, a4_source = read_pose_frames(
        root, path, "A4", camera, subject)
    train = [(slot, m, g) for slot, m, g in squat if slot < TRAIN_SLOTS]
    test = [(slot, m, g) for slot, m, g in squat if slot >= TRAIN_SLOTS]
    if min(len(train), len(test), len(deadlift)) < MIN_VALID_PER_SET:
        raise ValueError("Not enough independent held-out frames for a fixed alignment")
    rotation, scale = fit_fixed_similarity(
        np.stack([f[1] for f in train]),
        np.stack([f[2] for f in train]))
    return {
        "subject": subject, "camera": camera, "model": model,
        "fixed_similarity_fit_source": "A3 D2 first 12 sampled verified RGB frames",
        "fit_valid_frames": len(train),
        "fit_scale_WCS_native_per_model_meter": round(scale, 5),
        "A3_heldout": heldout_discrepancy(test, rotation, scale),
        "A4_cross_action": heldout_discrepancy(deadlift, rotation, scale),
        "source_alignment_audit": {"A3": a3_source, "A4": a4_source},
        "per_frame_data_or_fitted_matrix_saved": False,
    }


def aggregate(rows):
    """Five equal-weight calibration anchors, no pooled-frame pseudo-replication."""
    if len(rows) != len(ANCHORS) * len(CAMERAS) * len(VARIANTS):
        raise ValueError("Missing recordings or models")
    cells = []
    for camera in CAMERAS:
        for model in VARIANTS:
            group = [r for r in rows
                     if r["camera"] == camera and r["model"] == model]
            if len(group) != len(ANCHORS) or {r["subject"] for r in group} != set(ANCHORS):
                raise ValueError("Missing or duplicate calibration group")
            for label in ("A3_heldout", "A4_cross_action"):
                scores = [r[label][
                    "mean_discrepancy_pct_reference_femur_length"]
                          for r in group]
                cells.append({
                    "camera": camera, "model": model,
                    "evaluation_set": label,
                    "independent_calibration_groups": len(group),
                    "median_session_mean_6joint_discrepancy_pct_femur": round(
                        float(np.median(scores)), 4),
                    "range_session_mean_discrepancy_pct_femur": [
                        round(float(min(scores)), 4),
                        round(float(max(scores)), 4),
                    ],
                    "sum_valid_evaluation_frames": sum(
                        r[label]["valid_evaluation_frames"] for r in group),
                })
    return {
        "status": "complete_exploratory_root_relative_holdout",
        "metric": "A3-fixed-similarity root-centred 6-joint discrepancy / Vicon femur",
        "not_absolute_mpjpe_or_validated_mm_accuracy": True,
        "train": "A3 D2 first 12 unique verified images per camera/model/participant",
        "A3_test": "A3 D2 last 12 images, transform frozen",
        "A4_test": "A4 D2 all 24 images, transform frozen from A3",
        "total_recording_model_pairs": len(rows) * len(ACTIONS),
        "total_model_images_attempted": sum(
            r["source_alignment_audit"][act]["model_images_attempted"]
            for r in rows for act in ACTIONS),
        "total_pose_detections": sum(
            r["source_alignment_audit"][act]["pose_detections"]
            for r in rows for act in ACTIONS),
        "previous_sampling_invalid_coordinate_pairs": sum(
            rows[i]["source_alignment_audit"][act][
                "previous_index_selection_invalid_pairs"]
            for i in range(0, len(rows), len(VARIANTS))
            for act in ACTIONS),
        "independent_calibration_groups": len(ANCHORS),
        "new_independent_calibration_groups": 0,
        "cells": cells,
        "source_rgb_or_model_landmarks_published": False,
        "physical_sync_verified": False,
        "anatomical_joint_centre_equivalence_verified": False,
        "root_translation_and_camera_to_Vicon_registration_verified": False,
        "no_training_on_A4": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--detailed-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    source_dir = args.analysis_dir.resolve()
    paths = [args.detailed_output.resolve(), args.aggregate_output.resolve()]
    if paths[0] == paths[1] or any(
        path.exists() or path.parent != source_dir or
        path == repo or repo in path.parents for path in paths
    ):
        parser.error("Output must be two distinct new files outside repository in approved analysis directory")
    protocol = json.loads((source_dir /
        "A3_26subject_D2_census_protocol_20261116.json").read_text(encoding="utf8"))
    models = {}
    for model in VARIANTS:
        path = args.model_dir / f"pose_landmarker_{model}.task"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != protocol["model_variants"][model]["sha256"]:
            parser.error("Model SHA mismatch versus frozen A3 protocol")
        models[model] = path
    rows = [
        run_one(root, subject, camera, model, models[model])
        for subject in ANCHORS for camera in CAMERAS for model in VARIANTS
    ]
    overview = aggregate(rows)
    detailed = {
        "status": overview["status"], "model_sha256": {
            m: protocol["model_variants"][m]["sha256"] for m in VARIANTS},
        "rows": rows,
        "per_frame_xyz_or_images_saved": False,
        "private_local_only": True,
    }
    for path, payload in zip(paths, (detailed, overview)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
    print(json.dumps(overview, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
