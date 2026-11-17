"""Exploratory A3-fitted 3D knee-angle bias transfer to held-out A3/A4.

This is a *supervised Vicon-reference oracle* calibration audit, NOT an
anatomical landmark harmonization or deployable camera-only correction.
S1/S6/S11/S17/S23 represent FIVE historical calibration groups, not five
independent external validations. Never export source imagery or keypoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from .geometry import joint_angle
from .position_holdout import (
    ACTIONS, ANCHORS, CAMERAS, GT_JOINTS, MIN_VALID_PER_SET,
    MODEL_JOINTS, TRAIN_SLOTS, VARIANTS, read_pose_frames,
)

BANDS = ("deep_lt120", "mid_120_150", "extended_ge150")
SPLITS = ("A3_heldout", "A4_transfer")
CORRECTIONS = ("subject_A3_train", "leave_one_group_out_A3_train")


def _band(reference_angle: float) -> str:
    return BANDS[0] if reference_angle < 120 else (
        BANDS[1] if reference_angle < 150 else BANDS[2]
    )


def angle_observations(frames: list[tuple]) -> list[dict]:
    """In-memory left/right 3D included-angle pairs. No XYZ retained."""
    values = []
    for slot, model, reference in frames:
        for indices in ((0, 1, 2), (3, 4, 5)):
            mp_angle = joint_angle(*(model[i] for i in indices))
            gt_angle = joint_angle(*(reference[i] for i in indices))
            if not (math.isfinite(mp_angle) and math.isfinite(gt_angle)):
                raise ValueError("Invalid 3D joint included angle")
            values.append({"slot": int(slot), "model": float(mp_angle),
                           "vicon": float(gt_angle), "band": _band(gt_angle)})
    return values


def invariant_under_axis_flip(frame: tuple, *, atol=1e-9) -> bool:
    """Orthogonal axis sign/rotation changes cannot repair included angles."""
    _, model, reference = frame
    proper = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    mirrored = np.diag([-1., 1., 1.])
    for points in (model, reference):
        for idx in ((0, 1, 2), (3, 4, 5)):
            baseline = joint_angle(*(points[i] for i in idx))
            for matrix in (proper, mirrored):
                transformed = joint_angle(*(points[i] @ matrix for i in idx))
                if abs(transformed - baseline) > atol:
                    return False
    return True


def train_constant_bias(training: list[dict]) -> float:
    if len(training) < 2 * MIN_VALID_PER_SET:
        raise ValueError("Insufficient valid A3 calibration frames")
    return mean(x["model"] - x["vicon"] for x in training)


def evaluation_metrics(test: list[dict], bias: float) -> dict:
    """Never fit on test labels. Do not hide errors by clipping corrected angle."""
    if not test or not math.isfinite(bias):
        raise ValueError("Empty evaluation or invalid fixed bias")
    delta = [x["model"] - x["vicon"] for x in test]
    fixed = [error - bias for error in delta]
    def _entry(indices: list[int]) -> dict:
        base = [delta[i] for i in indices]
        corrected = [fixed[i] for i in indices]
        return {
            "valid_knee_angle_samples": len(indices),
            "baseline_mae_deg": round(mean(abs(v) for v in base), 6),
            "corrected_mae_deg": round(mean(abs(v) for v in corrected), 6),
            "mae_improvement_deg": round(
                mean(abs(v) for v in base) - mean(abs(v) for v in corrected), 6),
            "baseline_mean_signed_error_deg": round(mean(base), 6),
            "corrected_mean_signed_error_deg": round(mean(corrected), 6),
        }
    result = _entry(list(range(len(test))))
    result["by_vicon_included_angle"] = {
        band: (_entry([i for i, x in enumerate(test) if x["band"] == band])
               if any(x["band"] == band for x in test) else
               {"valid_knee_angle_samples": 0})
        for band in BANDS
    }
    result["corrected_angle_out_of_0_180_count"] = sum(
        not 0 <= x["model"] - bias <= 180 for x in test)
    return result


def _historical_map(analysis: Path) -> dict[tuple[str,str,str],dict]:
    prior = json.loads((analysis /
        "A3_A4_rootrelative_holdout_private_20261117.json").read_text(encoding="utf-8-sig"))
    index = {}
    for row in prior["rows"]:
        key = row["subject"], row["camera"], row["model"]
        if key in index:
            raise ValueError("Duplicate historical anchor/view/model")
        index[key] = row
    if len(index) != 20:
        raise ValueError("Expected exactly 20 prior A3/A4 anchor/view/model rows")
    return index


def evaluate_anchor(root: Path, subject: str, camera: str, variant: str,
                    model_asset: Path, prior: dict) -> dict:
    a3_frames, qa_a3 = read_pose_frames(
        root, model_asset, "A3", camera, subject)
    a4_frames, qa_a4 = read_pose_frames(
        root, model_asset, "A4", camera, subject)
    for action, quality in (("A3", qa_a3), ("A4", qa_a4)):
        old = prior["source_alignment_audit"][action]
        for field in (
            "model_images_attempted", "pose_detections", "valid_six_joint_frames",
            "previous_index_selection_invalid_pairs", "selected_jpg_names_different_from_previous",
        ):
            if quality[field] != old[field]:
                raise ValueError(
                    f"Prior frozen source-valid pairing replay failed: {action}/{field}")
        if (quality["ambiguous_or_unverifiable_jpg_groups"] or
                quality["selected_jpg_names_different_from_previous"] or
                quality["model_images_attempted"] != 24):
            raise ValueError("Strict 24 JPEG source-valid pairing not confirmed")
    train_frames = [f for f in a3_frames if f[0] < TRAIN_SLOTS]
    heldout_frames = [f for f in a3_frames if f[0] >= TRAIN_SLOTS]
    if (len(train_frames) != prior["fit_valid_frames"] or
            len(heldout_frames) != prior["A3_heldout"]["valid_evaluation_frames"] or
            len(a4_frames) != prior["A4_cross_action"]["valid_evaluation_frames"] or
            min(len(train_frames), len(heldout_frames), len(a4_frames)) < MIN_VALID_PER_SET):
        raise ValueError("Historical frame split or valid-count changed")
    if not all(invariant_under_axis_flip(f) for f in train_frames + heldout_frames + a4_frames):
        raise AssertionError("Orthogonal-invariance identity failed on actual data")
    obs_train = angle_observations(train_frames)
    obs_a3 = angle_observations(heldout_frames)
    obs_a4 = angle_observations(a4_frames)
    return {
        "subject": subject, "camera": camera, "model": variant,
        "a3_train_valid_frames": len(train_frames),
        "a3_holdout_valid_frames": len(heldout_frames),
        "a4_transfer_valid_frames": len(a4_frames),
        "a3_images_attempted": qa_a3["model_images_attempted"],
        "a4_images_attempted": qa_a4["model_images_attempted"],
        "a3_pose_detections": qa_a3["pose_detections"],
        "a4_pose_detections": qa_a4["pose_detections"],
        "original_middle_index_invalid_reference_A3": qa_a3[
            "previous_index_selection_invalid_pairs"],
        "original_middle_index_invalid_reference_A4": qa_a4[
            "previous_index_selection_invalid_pairs"],
        "a3_train_mean_signed_bias_deg": train_constant_bias(obs_train),
        "_A3_heldout": obs_a3, "_A4_transfer": obs_a4,
        "axis_rotation_or_reflection_changes_included_angle": False,
        "same_images_and_strict_references_as_prior_holdout": True,
    }


def make_summary(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Freeze A3 training biases, evaluate each historical group once per split."""
    wanted = {(s,c,m) for s in ANCHORS for c in CAMERAS for m in VARIANTS}
    if len(rows) != 20 or {(x["subject"], x["camera"], x["model"]) for x in rows} != wanted:
        raise ValueError("Expected all 20 unique anchor/camera/model combinations")
    details = []
    for row in rows:
        peers = [other["a3_train_mean_signed_bias_deg"] for other in rows
                 if other["subject"] != row["subject"] and
                 other["camera"] == row["camera"] and
                 other["model"] == row["model"]]
        if len(peers) != 4:
            raise ValueError("Leave-one-calibration-group-out requires four training anchors")
        for split in SPLITS:
            for method, fixed_bias in (
                (CORRECTIONS[0], row["a3_train_mean_signed_bias_deg"]),
                (CORRECTIONS[1], mean(peers)),
            ):
                metrics = evaluation_metrics(row["_" + split], fixed_bias)
                details.append({
                    "subject": row["subject"], "camera": row["camera"],
                    "model": row["model"], "test_split": split,
                    "calibration_method": method,
                    "fitted_A3_train_bias_deg": round(fixed_bias, 6),
                    "test_valid_frames": row["a3_holdout_valid_frames"] if split=="A3_heldout"
                    else row["a4_transfer_valid_frames"],
                    **metrics,
                })
    results = []
    for c in CAMERAS:
        for m in VARIANTS:
            for split in SPLITS:
                for method in CORRECTIONS:
                    selected = [r for r in details if
                        (r["camera"],r["model"],r["test_split"],r["calibration_method"])
                        == (c,m,split,method)]
                    if len(selected)!=5 or {r["subject"] for r in selected}!=set(ANCHORS):
                        raise ValueError("Incomplete 5-group comparison")
                    def _summary(rows, suffix=""):
                        if not rows:
                            return {"groups_with_data": 0}
                        return {
                            "groups_with_data":len(rows),
                            "knee_angle_samples":sum(r["valid_knee_angle_samples"] for r in rows),
                            "median_session_baseline_mae_deg":round(median(
                                r["baseline_mae_deg"] for r in rows),4),
                            "median_session_corrected_mae_deg":round(median(
                                r["corrected_mae_deg"] for r in rows),4),
                            "median_session_mae_improvement_deg":round(median(
                                r["mae_improvement_deg"] for r in rows),4),
                            "mean_session_mae_improvement_deg":round(mean(
                                r["mae_improvement_deg"] for r in rows),4),
                            "groups_improved":sum(r["mae_improvement_deg"]>0 for r in rows),
                            "groups_worsened":sum(r["mae_improvement_deg"]<0 for r in rows),
                        }
                    result = {
                        "camera":c, "model":m, "test_split":split,
                        "method":method, **_summary(selected),
                        "by_vicon_angle": {
                            band:_summary([
                                r["by_vicon_included_angle"][band] for r in selected
                                if r["by_vicon_included_angle"][band]["valid_knee_angle_samples"]
                            ]) for band in BANDS},
                        "post_correction_angles_outside_0_180_count":sum(
                            r["corrected_angle_out_of_0_180_count"] for r in selected),
                    }
                    results.append(result)
    overview = {
        "status":"complete_exploratory_bias_transfer",
        "question":"Does a constant Vicon-labelled A3 knee-angle bias transfer to withheld A3 and A4?",
        "anchor_groups":5, "new_independent_calibration_groups":0,
        "models":["lite","full"], "views":["C1","C6"],
        "training_data":"A3 D2 first 12 verified RGB slots only",
        "testing_data":["same-subject A3 last 12","same-subject A4 all 24"],
        "methods":{
            CORRECTIONS[0]:"One supervised per-subject A3 first-12 constant offset",
            CORRECTIONS[1]:"Mean A3 first-12 offsets from other 4 historical groups; target never fits",
        },
        "total_model_recording_pairs":40,
        "total_image_inferences":sum(x["a3_images_attempted"]+x["a4_images_attempted"]
                                     for x in rows),
        "total_pose_detections":sum(x["a3_pose_detections"]+x["a4_pose_detections"]
                                     for x in rows),
        "replayed_prior_strict_source_pairs_and_validity":True,
        "orthogonal_3d_rotation_and_reflection_leave_joint_angle_invariant":True,
        "results":results,
        "limits":{
            "bias_training_requires_Vicon_3D_ground_truth":True,
            "subject_specific_method_uses_Vicon_labels_from_same_target":True,
            "parity_transform_alone_cannot_change_included_knee_angle":True,
            "anatomical_joint_centre_equivalence_is_not_established":True,
            "raw_capture_time_synchronization_unverified":True,
            "heldout_test_groups_are_historical_not_external_new_calibrations":True,
            "sampling_first_half_may_have_different_motion_phase":True,
            "bias_correction_is_not_clinical_or_deployable":True,
            "no_model_retraining":True,
            "no_frame_landmarks_images_or_private_person_ids_in_public_aggregate":True,
        },
    }
    return overview, details


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    folder = args.analysis_dir.resolve()
    destinations = [args.private_output.resolve(), args.aggregate_output.resolve()]
    if (len(set(destinations))!=2 or any(
        path.exists() or path.parent!=folder or path==repo or repo in path.parents
        for path in destinations
    )):
        parser.error("Write only new distinct reports directly to authorized analysis dir")
    protocol = json.loads(
        (folder/"A3_new_subject_generalization_protocol_20261116.json"
        ).read_text(encoding="utf-8-sig"))
    assets = {}
    for variant in VARIANTS:
        path = args.model_dir/f"pose_landmarker_{variant}.task"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != protocol["models"][variant]["sha256"]:
            parser.error("Frozen model SHA256 mismatch")
        assets[variant] = path
    previous = _historical_map(folder)
    records = [evaluate_anchor(args.root, s, c, v, assets[v], previous[(s,c,v)])
               for s in ANCHORS for c in CAMERAS for v in VARIANTS]
    overview, evaluations = make_summary(records)
    private = {
        "status":overview["status"],
        "model_asset_sha256":{v:protocol["models"][v]["sha256"] for v in VARIANTS},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_summary_rows":[{
            key:value for key,value in row.items() if not key.startswith("_")
        } for row in records],
        "private_evaluation_rows":evaluations,
        "no_original_images_or_raw_model_keypoints_saved":True,
    }
    for target,report in zip(destinations,(private,overview)):
        with target.open("x",encoding="utf8") as f:
            json.dump(report,f,ensure_ascii=False,indent=2)
            f.write("\n")
    print(json.dumps({
        "status":overview["status"],
        "model_recording_pairs":overview["total_model_recording_pairs"],
        "image_inferences":overview["total_image_inferences"],
        "pose_detections":overview["total_pose_detections"],
        "cells":[{k:cell[k] for k in (
            "camera","model","test_split","method",
            "median_session_baseline_mae_deg","median_session_corrected_mae_deg",
            "median_session_mae_improvement_deg","groups_improved","groups_worsened")}
            for cell in overview["results"]],
    }))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
