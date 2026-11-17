"""MediaPipe-to-Vicon root-relative coordinate parity diagnostic (not MPJPE).

Fit proper and reflected similarity transforms on exactly the same A3-first-12
inferred joint frames. Freeze both fits for A3 held-out and cross-action A4.
The reflected transform diagnoses numeric coordinate convention; it is not
a physical rigid camera rotation nor proof of the model's true handedness.
Vicon GT hips are used to root-centre each evaluation frame.

All model-image input, per-person reports, fitted matrices and participant
identifiers remain in approved local private storage, never public Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from .position_holdout import (
    ANCHORS, CAMERAS, VARIANTS, ACTIONS, TRAIN_SLOTS, MIN_VALID_PER_SET,
    heldout_discrepancy, read_pose_frames
)

TOLERANCE_PERCENT = 0.001


def fit_parity_similarity(model_train: np.ndarray, vicon_train: np.ndarray,
                          *, determinant: int):
    """Fitted train-only scale/orthogonal matrix with explicit determinant."""
    if determinant not in (1, -1):
        raise ValueError("determinant must be +1 (proper) or -1 (reflected)")
    x, y = np.asarray(model_train, dtype=float), np.asarray(vicon_train, dtype=float)
    if (x.ndim != 3 or x.shape[1:] != (6, 3)
            or y.shape != x.shape or len(x) < MIN_VALID_PER_SET):
        raise ValueError("Require >=6 paired 6-joint A3 train frames")
    a, b = x.reshape(-1, 3), y.reshape(-1, 3)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite model or Vicon train points")
    denom = float(np.sum(a*a))
    if denom <= 1e-8:
        raise ValueError("Degenerate 3D train model cloud")
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.eye(3)
    d[2, 2] = determinant * np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = float(np.sum((a @ rotation) * b) / denom)
    if (not np.isfinite(scale) or scale <= 0 or
            abs(np.linalg.det(rotation) - determinant) > 1e-5):
        raise ValueError("Invalid parity-constrained alignment")
    return rotation, scale


def compare_holdout(train, a3_holdout, a4_holdout):
    """Never select parity based on test scores; report both frozen fits."""
    if min(len(train), len(a3_holdout), len(a4_holdout)) < MIN_VALID_PER_SET:
        raise ValueError("Insufficient A3 train/A3 holdout/A4 test six-joint frames")
    x = np.stack([r[1] for r in train])
    y = np.stack([r[2] for r in train])
    fits = {}
    for parity, det in (("proper", 1), ("reflected", -1)):
        rotation, scale = fit_parity_similarity(x, y, determinant=det)
        fits[parity] = {
            "determinant": det,
            "fit_scale_WCS_native_per_model_m": round(scale, 5),
            "A3_heldout": heldout_discrepancy(a3_holdout, rotation, scale),
            "A4_cross_action": heldout_discrepancy(a4_holdout, rotation, scale),
        }
    return fits


def one_recording(root: Path, subject: str, camera: str,
                  model_variant: str, model_path: Path, prior: dict):
    """Run the same 24 JPGs for two parity hypotheses; nothing saved per frame."""
    squat, a3_source = read_pose_frames(
        root, model_path, "A3", camera, subject)
    deadlift, a4_source = read_pose_frames(
        root, model_path, "A4", camera, subject)
    train = [r for r in squat if r[0] < TRAIN_SLOTS]
    a3_test = [r for r in squat if r[0] >= TRAIN_SLOTS]
    fits = compare_holdout(train, a3_test, deadlift)
    old_proper = prior["A3_heldout"]
    old_a4 = prior["A4_cross_action"]
    # Previous pipeline used the same SVD fitter for determinant +1.
    for name, old in (("A3_heldout", old_proper), ("A4_cross_action", old_a4)):
        got = fits["proper"][name]
        if (got["valid_evaluation_frames"] != old["valid_evaluation_frames"] or
            abs(got["mean_discrepancy_pct_reference_femur_length"] -
                old["mean_discrepancy_pct_reference_femur_length"]) > TOLERANCE_PERCENT):
            raise ValueError("Prior proper-rotation baseline could not be replayed")
    for act, q in (("A3", a3_source), ("A4", a4_source)):
        old_q = prior["source_alignment_audit"][act]
        for field in ("model_images_attempted", "pose_detections",
                      "valid_six_joint_frames", "selected_jpg_names_different_from_previous"):
            if q[field] != old_q[field]:
                raise ValueError("Frozen source or model detection changed")
    return {
        "subject": subject,
        "camera": camera,
        "model": model_variant,
        "a3_fit_frames": len(train),
        "strict_source_frames_and_frozen_model_reproduced": True,
        "prior_proper_result_independently_replayed": True,
        "A3_source_audit": a3_source,
        "A4_source_audit": a4_source,
        "fits": fits,
        "test_frames_used_for_fit": False,
        "Vicon_GТ_hip_root_used_in_evaluation": True,
        "per_frame_model_or_reference_coordinates_saved": False,
        "physical_mpjpe_mm_validated": False,
    }


def aggregate(rows: list[dict]) -> dict:
    if len(rows) != len(ANCHORS) * len(CAMERAS) * len(VARIANTS):
        raise ValueError("Expected exactly 20 anchor/camera/model comparisons")
    expected = {(s, c, m) for s in ANCHORS for c in CAMERAS for m in VARIANTS}
    if {(r["subject"], r["camera"], r["model"]) for r in rows} != expected:
        raise ValueError("Missing/duplicate calibration anchor or model")
    cells = []
    all_stability = []
    for camera in CAMERAS:
        for model in VARIANTS:
            subset = [r for r in rows
                      if r["camera"] == camera and r["model"] == model]
            stable = []
            for r in subset:
                proper_a3 = r["fits"]["proper"]["A3_heldout"][
                    "mean_discrepancy_pct_reference_femur_length"]
                reflect_a3 = r["fits"]["reflected"]["A3_heldout"][
                    "mean_discrepancy_pct_reference_femur_length"]
                proper_a4 = r["fits"]["proper"]["A4_cross_action"][
                    "mean_discrepancy_pct_reference_femur_length"]
                reflect_a4 = r["fits"]["reflected"]["A4_cross_action"][
                    "mean_discrepancy_pct_reference_femur_length"]
                stable.append((reflect_a3 < proper_a3) ==
                              (reflect_a4 < proper_a4))
            all_stability.extend(stable)
            for label in ("A3_heldout", "A4_cross_action"):
                p = [r["fits"]["proper"][label][
                    "mean_discrepancy_pct_reference_femur_length"]
                     for r in subset]
                mirrored = [r["fits"]["reflected"][label][
                    "mean_discrepancy_pct_reference_femur_length"]
                            for r in subset]
                improv = [a-b for a,b in zip(p,mirrored)]
                cells.append({
                    "camera": camera, "model": model,
                    "evaluation_set": label,
                    "independent_original_calibration_groups": 5,
                    "median_group_mean_proper_pct_femur": round(float(median(p)), 4),
                    "median_group_mean_reflected_pct_femur": round(float(median(mirrored)), 4),
                    "median_paired_proper_minus_reflected_pct_points": round(float(median(improv)), 4),
                    "reflected_better_group_count": sum(b<a for a,b in zip(p,mirrored)),
                    "proper_better_group_count": sum(a<b for a,b in zip(p,mirrored)),
                    "ties": sum(a==b for a,b in zip(p,mirrored)),
                    "valid_six_joint_test_frames": sum(
                        r["fits"]["proper"][label]["valid_evaluation_frames"]
                        for r in subset),
                })
    return {
        "status": "complete_exploratory_mediapipe_parity_heldout",
        "study": "A3-fit MediaPipe hip-centred world landmarks vs Vicon WCS hip-centred joints, same five anchors",
        "train": "A3 D2 first 12 source-valid RGB images; two fixed fits (+1/-1 determinant)",
        "test_sets": ["A3 D2 last 12 frames", "A4 D2 24 source-valid frames"],
        "model_report_pairs": len(rows),
        "model_image_inferences": sum(
            r[f"{a}_source_audit"]["model_images_attempted"]
            for r in rows for a in ACTIONS),
        "pose_detections": sum(
            r[f"{a}_source_audit"]["pose_detections"]
            for r in rows for a in ACTIONS),
        "independent_calibration_groups": 5,
        "new_independent_calibration_groups": 0,
        "paired_parity_preference_stable_A3_to_A4_count": sum(all_stability),
        "paired_parity_preference_total": len(all_stability),
        "cells": cells,
        "absolute_world_camera_registration_verified": False,
        "physical_camera_Vicon_exposure_sync_verified": False,
        "Vicon_and_MediaPipe_anatomical_joint_centres_equivalent": False,
        "framewise_Vicon_GT_hip_centre_used_to_root_relative": True,
        "absolute_MPJPE_mm_validated": False,
        "per_frame_xyz_images_or_fits_published": False,
        "clinical_or_causal_claim_permitted": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-new", type=int, default=20)
    args = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    folder = args.analysis_dir.resolve()
    out = args.output_dir.resolve()
    if (out != folder / "mediapipe_parity_A3_A4_20261117" or
            out == repo or repo in out.parents or not 1 <= args.max_new <= 20):
        p.error("Output must be dedicated private local analysis subfolder")
    protocol = json.loads((folder / "A3_26subject_D2_census_protocol_20261116.json").read_text(encoding="utf-8-sig"))
    previous = json.loads((folder / "A3_A4_rootrelative_holdout_private_20261117.json").read_text(encoding="utf-8-sig"))
    previous_rows = {(r["subject"], r["camera"], r["model"]):r
                     for r in previous["rows"]}
    if len(previous_rows) != 20 or previous["per_frame_xyz_or_images_saved"] is not False:
        raise ValueError("Unexpected original private cross-action baseline")
    models = {}
    for model in VARIANTS:
        model_path = args.model_dir / f"pose_landmarker_{model}.task"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if (digest != protocol["model_variants"][model]["sha256"] or
                digest != previous["model_sha256"][model]):
            raise ValueError("Frozen MediaPipe Lite/Full model asset SHA changed")
        models[model] = model_path
    out.mkdir(parents=True, exist_ok=True)
    rows, computed = [], 0
    for sub in ANCHORS:
        for camera in CAMERAS:
            for model in VARIANTS:
                key = sub,camera,model
                file = out / f"parity_{sub}_{camera}_{model}.json"
                if file.is_file():
                    row=json.loads(file.read_text(encoding="utf-8-sig"))
                elif computed < args.max_new:
                    row = one_recording(args.root, sub, camera, model,
                                        models[model], previous_rows[key])
                    with file.open("x", encoding="utf-8") as h:
                        json.dump(row, h, ensure_ascii=False, indent=2)
                        h.write("\n")
                    computed += 1
                else:
                    continue
                if (row["subject"],row["camera"],row["model"])!=key:
                    raise ValueError("Private cached report identity mismatch")
                if not row["prior_proper_result_independently_replayed"]:
                    raise ValueError("Missing baseline replay validation")
                rows.append(row)
    if len(rows)!=20:
        print(json.dumps({"status":"incomplete_parity_recheck",
                          "completed_reports":len(rows),
                          "required_reports":20,
                          "new_reports_this_run":computed}))
        return 0
    result=aggregate(rows)
    dest=out/"mediapipe_A3A4_parity_aggregate_20261117.json"
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8-sig"))!=result:
            raise ValueError("Conflicting previous summary, preserving")
    else:
        with dest.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
            f.write("\n")
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
