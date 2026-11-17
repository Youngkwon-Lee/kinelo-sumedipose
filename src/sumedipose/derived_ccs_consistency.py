"""Derived Vicon WCS-to-camera XYZ consistency and coordinate-convention audit.

Both input coordinate arrays originate from Vicon. This CANNOT validate the
physical camera's independent 3D registration, real-world MPJPE or exposure
time synchronization. Compare proper vs reflected orthogonal transforms.
Original Vicon/RGB data and per-person results are private and never exported
to the public repository.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import numpy as np

from .calibration_strata import SESSIONS, CAMERAS
from .io import read_sample
from .position_holdout import GT_JOINTS, strict_unique_pairs
from .sync_evidence import _timestamp_fields

ACTION = "A3"
HELDOUT_ACTION = "A4"
REPEAT = "D2"
SAMPLED = 24
TRAIN = 12
SEGMENTS = ((0, 1), (1, 2), (3, 4), (4, 5))


def orthogonal_similarity(xyz_wcs: np.ndarray, xyz_ccs: np.ndarray, *,
                          determinant: int):
    """Fit fixed scale, rotation/orientation, and translation using A3 training data.

    determinant=+1 is a proper physical rotation. determinant=-1 is an
    improper orthogonal matrix, a possible coordinate-axis handedness
    convention diagnostic; NOT a physical rotation.
    """
    if determinant not in (-1, 1):
        raise ValueError("Determinant convention must be +1 or -1")
    x = np.asarray(xyz_wcs, dtype=float)
    y = np.asarray(xyz_ccs, dtype=float)
    if x.shape != y.shape or x.ndim != 3 or x.shape[1:] != (6, 3) or len(x) < 6:
        raise ValueError("Expected at least six paired 6-by-3 training frames")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite coordinate input")
    a, b = x.reshape(-1, 3), y.reshape(-1, 3)
    center_a, center_b = a.mean(axis=0), b.mean(axis=0)
    ax, by = a - center_a, b - center_b
    norm = float(np.sum(ax * ax))
    if norm <= 1e-12:
        raise ValueError("Degenerate source point cloud")
    u, _, vt = np.linalg.svd(ax.T @ by)
    d = np.eye(3)
    d[2, 2] = determinant * np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = float(np.sum((ax @ rotation) * by) / norm)
    if (scale <= 0 or not math.isfinite(scale)
            or abs(np.linalg.det(rotation) - determinant) > 1e-5):
        raise ValueError("Invalid orthogonal similarity fit")
    translation = center_b - scale * (center_a @ rotation)
    return rotation, scale, translation


def paired_error(wcs: np.ndarray, ccs: np.ndarray, fit: tuple):
    """Compare two Vicon-derived representations; no MediaPipe or MPJPE."""
    r, scale, translation = fit
    points = np.asarray(wcs, float)
    target = np.asarray(ccs, float)
    if points.shape != target.shape or points.ndim != 3 or points.shape[1:] != (6, 3):
        raise ValueError("Invalid held-out reference shape")
    if not np.isfinite(points).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite held-out coordinate")
    errs_in_wcs_native = np.linalg.norm(
        scale * (points @ r) + translation - target, axis=2) / scale
    femur = (
        np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
        + np.linalg.norm(points[:, 3] - points[:, 4], axis=1)
    ) / 2
    if np.any(femur < 1e-8) or not np.isfinite(femur).all():
        raise ValueError("Invalid reference femur segment")
    return {
        "mean_discrepancy_WCS_native_units": round(
            float(np.mean(errs_in_wcs_native)), 5),
        "mean_discrepancy_pct_reference_femur": round(
            float(np.mean(100 * np.mean(errs_in_wcs_native, axis=1) / femur)), 5),
        "evaluated_frames": len(points),
    }


def _sample_source(root: Path, subject: str, camera: str, action: str):
    paths, world, records = read_sample(
        root, camera=camera, subject=subject, action=action, repeat=REPEAT)
    if len(world) != len(records) or not world:
        raise ValueError("Missing or misaligned original Vicon coordinate records")
    if world[0]["point_ids"] != records[0]["point_ids"]:
        raise ValueError("WCS and camera joint labels disagree")
    w_indices = [world[0]["point_ids"].index(j) for j in GT_JOINTS]
    c_indices = [records[0]["point_ids"].index(j) for j in GT_JOINTS]
    with ZipFile(paths["rgb"]) as archive:
        picks, qa = strict_unique_pairs(records, archive.namelist(), SAMPLED)
    wcs, ccs = [], []
    for _, idx in picks:
        x = np.asarray(world[idx]["xyz"], float)[w_indices]
        y = np.asarray(records[idx]["xyz"], float)[c_indices]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite data in a frozen source RGB sample")
        wcs.append(x)
        ccs.append(y)
    wcs, ccs = np.stack(wcs), np.stack(ccs)
    lengths_ratio = []
    for aa, bb in SEGMENTS:
        xlen = np.linalg.norm(wcs[:, aa] - wcs[:, bb], axis=1)
        ylen = np.linalg.norm(ccs[:, aa] - ccs[:, bb], axis=1)
        if np.any(ylen <= 1e-8):
            raise ValueError("Zero camera-derived reference segment length")
        lengths_ratio.extend((xlen / ylen).tolist())
    return wcs, ccs, {
        "strict_rgb_pairs": qa["unique_verified_jpgs_selected"],
        "old_wrong_coordinate_index_count": qa[
            "previous_index_selection_invalid_pairs"],
        "source_wcs_to_ccs_segment_length_ratio_median": round(
            float(median(lengths_ratio)), 6),
        "original_WCS_timestamp_keys": _timestamp_fields(world[0]),
        "original_camera_timestamp_keys": _timestamp_fields(records[0]),
    }


def run_subject(root: Path, subject: str, camera: str, *, include_a4: bool):
    a3_w, a3_c, qa_a3 = _sample_source(root, subject, camera, ACTION)
    qa_a4 = None
    if include_a4:
        a4_w, a4_c, qa_a4 = _sample_source(
            root, subject, camera, HELDOUT_ACTION)
    outputs = {}
    for sign, kind in ((1, "proper_rotation"), (-1, "axis_reversal_allowed")):
        fit = orthogonal_similarity(
            a3_w[:TRAIN], a3_c[:TRAIN], determinant=sign)
        outputs[kind] = {
            "fit_scale_CCS_units_per_WCS_native": round(fit[1], 9),
            "fit_matrix_determinant_sign": sign,
            "A3_heldout": paired_error(a3_w[TRAIN:], a3_c[TRAIN:], fit),
            **({"A4_transfer": paired_error(a4_w, a4_c, fit)}
               if include_a4 else {}),
        }
    return {
        "subject": subject,
        "camera": camera,
        "A3": qa_a3,
        "A4": qa_a4,
        "comparison": outputs,
        "fitted_on_A3_first_12_only": True,
        "camera_XYZ_is_derived_from_same_Vicon_source": True,
        "physical_camera_to_Vicon_registration_independently_verified": False,
        "timestamp_alignment_independently_verified": False,
        "anatomical_joint_definitions_harmonized_with_MediaPipe": False,
    }


def aggregate(rows: list[dict], subject_map: dict[str, str]) -> dict:
    expected = {f"S{i}" for i in range(1, 29) if i not in (5, 10)}
    keys = {(r["subject"], r["camera"]) for r in rows}
    if (len(rows) != 52 or len(keys) != 52 or
            keys != {(s, c) for s in expected for c in CAMERAS}
            or {subject_map[s] for s in expected} != set(SESSIONS)):
        raise ValueError("Expected original 26 participants and two views, five calibration groups")
    findings = []
    for camera in CAMERAS:
        selected = [r for r in rows if r["camera"] == camera]
        for scope, label in (("A3_heldout", "A3_squat"),
                             ("A4_transfer", "A4_deadlift")):
            subset = [
                r for r in selected if scope in
                r["comparison"]["proper_rotation"]
            ]
            if len(subset) != (26 if scope == "A3_heldout" else 5):
                raise ValueError("Missing original subject sets in action comparison")
            group_rows = {s: [r for r in subset if subject_map[r["subject"]] == s]
                          for s in SESSIONS}
            if not all(group_rows.values()):
                raise ValueError("Missing a calibration group")
            if scope == "A4_transfer" and any(
                len(group_rows[s]) != 1 for s in SESSIONS):
                raise ValueError("A4 must use one anchor participant per group")
            kinds = {}
            for k in ("proper_rotation", "axis_reversal_allowed"):
                unit = "mean_discrepancy_WCS_native_units"
                pct = "mean_discrepancy_pct_reference_femur"
                by_group = [
                    mean(r["comparison"][k][scope][unit] for r in group_rows[s])
                    for s in SESSIONS
                ]
                by_group_pct = [
                    mean(r["comparison"][k][scope][pct] for r in group_rows[s])
                    for s in SESSIONS
                ]
                kinds[k] = {
                    "median_five_equal_weight_group_mean_WCS_native": round(
                        median(by_group), 5),
                    "median_five_equal_weight_group_mean_pct_femur": round(
                        median(by_group_pct), 5),
                    "range_group_mean_WCS_native": [
                        round(min(by_group), 5), round(max(by_group), 5)
                    ],
                }
            reflected_better_count = sum(
                r["comparison"]["axis_reversal_allowed"][scope][
                    "mean_discrepancy_WCS_native_units"
                ] < r["comparison"]["proper_rotation"][scope][
                    "mean_discrepancy_WCS_native_units"
                ] for r in subset
            )
            findings.append({
                "camera": camera,
                "evaluation_action": label,
                "source_subject_records": len(subset),
                "distinct_original_calibration_groups": len(SESSIONS),
                "reflected_fit_better_in_subject_camera_cases":
                    reflected_better_count,
                "fits": kinds,
            })
    ratios = [r["A3"][
        "source_wcs_to_ccs_segment_length_ratio_median"] for r in rows]
    scale = [r["comparison"]["axis_reversal_allowed"][
        "fit_scale_CCS_units_per_WCS_native"] for r in rows]
    missing_wcs = sum(not r["A3"]["original_WCS_timestamp_keys"] for r in rows)
    missing_camera = sum(not r["A3"]["original_camera_timestamp_keys"] for r in rows)
    return {
        "status": "complete_derived_Vicon_coordinate_consistency_audit",
        "study": "SUMediPose D2 A3 full 26 and A4 five anchors Vicon WCS-to-derived-CCS integrity",
        "camera_views": list(CAMERAS),
        "A3_participants_checked": 26,
        "A4_anchor_participants_checked": 5,
        "A3_view_recordings": 52,
        "A4_view_recordings": 10,
        "A3_training_frames_per_view_participant": TRAIN,
        "A3_holdout_frames_per_view_participant": SAMPLED - TRAIN,
        "A4_holdout_frames_per_view_anchor": SAMPLED,
        "independent_calibration_groups": len(SESSIONS),
        "new_independent_calibration_groups": 0,
        "WCS_to_derived_CCS_segment_length_ratio_median": round(median(ratios), 6),
        "reflected_fit_CCS_per_WCS_scale_median": round(median(scale), 9),
        "A3_WCS_records_without_timestamp_keys": missing_wcs,
        "A3_derived_CCS_records_without_timestamp_keys": missing_camera,
        "group_results": findings,
        "no_absolute_camera_or_model_registration_verified": True,
        "no_physical_camera_Vicon_common_clock_evidence": True,
        "model_anatomical_landmark_equivalence_verified": False,
        "physical_mpjpe_mm_validated": False,
        "fitted_transform_is_derived_source_self_consistency_not_independent_reference": True,
        "fitted_scale_reflection_is_not_physical_rigid_camera_extrinsic": True,
        "raw_xyz_or_subject_data_exported": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--aggregate-report", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    analysis = args.analysis_dir.resolve()
    private_path = args.private_report.resolve()
    group_path = args.aggregate_report.resolve()
    repo = Path(__file__).resolve().parents[2]
    if (private_path == group_path or
            any(p.exists() or p.parent != analysis or
                p == repo or repo in p.parents
                for p in (private_path, group_path))):
        parser.error("Refuse overwrite or output outside approved private analysis directory")
    groupmap = json.loads((root / "calibration" / "external" /
                           "subject_map.json").read_text(encoding="utf-8-sig"))
    subjects = [f"S{i}" for i in range(1, 29) if i not in (5, 10)]
    rows = [
        run_subject(root, subject, cam, include_a4=subject in SESSIONS)
        for subject in subjects for cam in CAMERAS
    ]
    overview = aggregate(rows, groupmap)
    detailed = {
        "status": overview["status"], "rows": rows,
        "original_xyz_positions_or_matrices_saved": False,
        "private_local_only": True,
    }
    for p, obj in ((private_path, detailed), (group_path, overview)):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps({
        "status": overview["status"],
        "A3_participants_checked": overview["A3_participants_checked"],
        "A4_anchor_participants_checked": overview["A4_anchor_participants_checked"],
        "independent_calibration_groups": overview["independent_calibration_groups"],
        "physical_mpjpe_mm_validated": overview["physical_mpjpe_mm_validated"],
        "group_results": overview["group_results"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
