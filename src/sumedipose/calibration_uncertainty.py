"""Local-only checkerboard lens and stereo-extrinsic uncertainty audit.

All statistics are aggregates from non-fit frames. Per-camera board poses
are inferred from their own images; source R/T/K remains frozen.
This does not establish independent Vicon 3D accuracy or capture timestamps.
"""
from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .heldout_calibration import (
    _corners, _frame_ids, _fundamental, _symmetry,
    audit_stereo_pair, sampson_distance_pixels,
)


def _summarize(values):
    if not values:
        return {"n": 0, "median": None, "p90": None}
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite calibration statistic")
    return {"n": len(a), "median": round(float(np.median(a)), 5),
            "p90": round(float(np.percentile(a, 90)), 5)}


def relative_pose_disagreement(Ra, ta, Rb, tb, Rsource, Tsource):
    """Rotation degrees and translation difference in source chessboard squares.

    Poses Ra/ta and Rb/tb are estimated independently from the same planar
    checkerboard. This check is not external metric world ground truth.
    """
    Ra, Rb, Rsource = (np.asarray(r, dtype=float).reshape(3, 3)
                      for r in (Ra, Rb, Rsource))
    ta, tb, Tsource = (np.asarray(t, dtype=float).reshape(3, 1)
                       for t in (ta, tb, Tsource))
    relative_r = Rb @ Ra.T
    relative_t = tb - relative_r @ ta
    cosine = np.clip((np.trace(relative_r @ Rsource.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine))), float(
        np.linalg.norm(relative_t - Tsource)
    )


def _raw_sampson(pts_a, pts_b, params):
    F = _fundamental(params)
    a = np.column_stack((pts_a, np.ones(len(pts_a))))
    b = np.column_stack((pts_b, np.ones(len(pts_b))))
    la = (F @ a.T).T
    lb = (F.T @ b.T).T
    return np.sqrt(np.sum(b * la, axis=1)**2 / np.maximum(
        np.sum(la[:, :2]**2, axis=1)+np.sum(lb[:, :2]**2, axis=1), 1e-30))


def stereo_pair_uncertainty(raw_root, external_zip, subject="S1", paired_camera="5"):
    """Fit-defined checkerboard orientation, then fixed-source holdout diagnostics.

    PnP fits the known 8x6 checkerboard to each camera separately and is an
    in-frame quality control (not independent validation of intrinsics).
    Transferring that first-camera pose with frozen stereo R/T tests whether
    the second-camera observations remain geometrically consistent.
    """
    import cv2
    raw_root = Path(raw_root)
    paired_camera = str(paired_camera).removeprefix("C")
    base = audit_stereo_pair(raw_root, external_zip, subject, paired_camera)
    with (raw_root / "calibration/external_calibration.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = [r for r in csv.DictReader(f)
                if r["subject"] == subject and r["cam"] == paired_camera]
    if len(rows) != 1:
        raise ValueError("Exactly one source camera pair row required")
    row = rows[0]
    fit_a, fit_b = _frame_ids(row["valid_frames_A"]), _frame_ids(row["valid_frames_B"])
    if fit_a != fit_b:
        raise ValueError("Source fit frame sets differ")
    rel = Path(row["path"].replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:3] != (
        "calibration", "external", "matrix"
    ):
        raise ValueError("Unsafe source matrix path")
    with np.load(raw_root / rel, allow_pickle=False) as z:
        params = {k: z[k] for k in ("R", "T", "mtx_A", "mtx_B", "dist_A", "dist_B")}

    obj = np.zeros((48, 3), dtype=float)
    obj[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    Rsource, Tsource = params["R"], params["T"].reshape(3, 1)
    K1, K2 = params["mtx_A"], params["mtx_B"]
    d1, d2 = params["dist_A"], params["dist_B"]
    group = base["source_group"]
    first_name = f"C{row['main']}{subject}_{group}"
    second_name = f"C{row['pair']}{subject}_{group}"
    fields = (
        "source_sampson_undistorted_median_px",
        "source_sampson_raw_undistorted_skipped_median_px",
        "image_lens_undistortion_shift_A_median_px",
        "image_lens_undistortion_shift_B_median_px",
        "single_camera_PnP_A_reprojection_median_px",
        "single_camera_PnP_B_reprojection_median_px",
        "source_stereo_R_T_cross_camera_reprojection_median_px",
        "independently_fit_board_relative_rotation_difference_degrees",
        "independently_fit_board_relative_translation_difference_square_units",
    )
    stats = {k: [] for k in fields}
    detections = 0
    with ZipFile(external_zip) as archive:
        names = set(archive.namelist())
        for index in range(1, 61):
            if index in fit_a:
                continue
            left = f"{first_name}/{first_name}_{index:04d}.jpg"
            right = f"{second_name}/{second_name}_{index:04d}.jpg"
            if left not in names or right not in names:
                continue
            A = _corners(archive, left)
            B = _corners(archive, right)
            if A is None or B is None:
                continue
            B = _symmetry(B, base["training_selected_symmetry"])
            A = np.asarray(A, np.float64)
            B = np.asarray(B, np.float64)
            detections += 1
            stats["source_sampson_undistorted_median_px"].append(
                float(np.median(sampson_distance_pixels(A, B, params))))
            stats["source_sampson_raw_undistorted_skipped_median_px"].append(
                float(np.median(_raw_sampson(A, B, params))))
            au = cv2.undistortPoints(A.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1,2)
            bu = cv2.undistortPoints(B.reshape(-1, 1, 2), K2, d2, P=K2).reshape(-1,2)
            stats["image_lens_undistortion_shift_A_median_px"].append(
                float(np.median(np.linalg.norm(au-A, axis=1))))
            stats["image_lens_undistortion_shift_B_median_px"].append(
                float(np.median(np.linalg.norm(bu-B, axis=1))))
            valid_a, rA, tA = cv2.solvePnP(obj, A, K1, d1, flags=cv2.SOLVEPNP_ITERATIVE)
            valid_b, rB, tB = cv2.solvePnP(obj, B, K2, d2, flags=cv2.SOLVEPNP_ITERATIVE)
            if not valid_a or not valid_b:
                continue
            Ra = cv2.Rodrigues(rA)[0]
            Rb = cv2.Rodrigues(rB)[0]
            predA, _ = cv2.projectPoints(obj, rA, tA, K1, d1)
            predB, _ = cv2.projectPoints(obj, rB, tB, K2, d2)
            stats["single_camera_PnP_A_reprojection_median_px"].append(
                float(np.median(np.linalg.norm(predA.reshape(-1, 2)-A, axis=1))))
            stats["single_camera_PnP_B_reprojection_median_px"].append(
                float(np.median(np.linalg.norm(predB.reshape(-1, 2)-B, axis=1))))
            Xb = (Rsource @ (Ra @ obj.T+tA)+Tsource).T
            cross, _ = cv2.projectPoints(
                Xb, np.zeros(3), np.zeros(3), K2, d2)
            stats["source_stereo_R_T_cross_camera_reprojection_median_px"].append(
                float(np.median(np.linalg.norm(cross.reshape(-1,2)-B,axis=1))))
            rotation_deg, trans_delta = relative_pose_disagreement(
                Ra, tA, Rb, tB, Rsource, Tsource
            )
            stats["independently_fit_board_relative_rotation_difference_degrees"].append(
                rotation_deg)
            stats["independently_fit_board_relative_translation_difference_square_units"].append(
                trans_delta)
    return {
        "session": subject, "source_group": group,
        "original_source_calibration_fixed": True,
        "source_fit_frame_ids_excluded": len(fit_a),
        "holdout_frames_declared": 60-len(fit_a),
        "holdout_frames_with_detected_paired_chessboards": detections,
        "source_checkerboard_world_scaling": 1,
        "square_physical_size_mm_known": False,
        "metrics": {key: _summarize(value) for key,value in stats.items()},
        "physical_exposure_clock_sync_verified": False,
        "independent_camera_world_3d_accuracy_verified": False,
        "notes": (
            "Single-camera board pose is fit to the SAME points used for its "
            "reprojection (not an intrinsic heldout test). Cross-camera reprojection "
            "retains original source stereo R/T. Relative pose comparisons use "
            "an 8x6 board with square length=1 and do not establish real mm. "
            "Differences may arise from calibration, asynchronous exposures, "
            "board detection/orientation, rolling shutter or motion."
        ),
    }
