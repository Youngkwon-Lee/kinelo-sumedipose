"""Held-out checkerboard stereo geometry audit (source calibration JPGs remain in ZIP).

The metric is symmetric Sampson epipolar residual in undistorted image pixels.
It is NOT 3D reprojection error, camera/Vicon clock sync, or clinical accuracy.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np


def _frame_ids(source: str) -> set[int]:
    items = ast.literal_eval(source)
    if not isinstance(items, list):
        raise ValueError("Expected source calibration frame path list")
    matches = [re.search(r"_(\d+)\.jpg$", str(item)) for item in items]
    if any(match is None for match in matches):
        raise ValueError("Calibration frame naming changed")
    return {int(match.group(1)) for match in matches}


def _symmetry(points: np.ndarray, name: str) -> np.ndarray:
    grid = points.reshape(6, 8, 2)
    if name == "same":
        return grid.reshape(-1, 2)
    if name == "flip_horizontal":
        return grid[:, ::-1].reshape(-1, 2)
    if name == "flip_vertical":
        return grid[::-1, :].reshape(-1, 2)
    if name == "rotate_180":
        return grid[::-1, ::-1].reshape(-1, 2)
    raise ValueError("Unknown checkerboard orientation")


def _fundamental(params: dict) -> np.ndarray:
    A, B = params["mtx_A"], params["mtx_B"]
    R, t = params["R"], params["T"].ravel()
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(B).T @ skew @ R @ np.linalg.inv(A)


def sampson_distance_pixels(pts_a, pts_b, params) -> np.ndarray:
    """Pixel Sampson distance after lens undistortion and source stereo R/T."""
    import cv2
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 1, 2)
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 1, 2)
    if a.shape != b.shape or not len(a):
        raise ValueError("Stereo point arrays must have equal nonzero size")
    A, B = params["mtx_A"], params["mtx_B"]
    a = cv2.undistortPoints(a, A, params["dist_A"], P=A).reshape(-1, 2)
    b = cv2.undistortPoints(b, B, params["dist_B"], P=B).reshape(-1, 2)
    aa = np.column_stack((a, np.ones(len(a))))
    bb = np.column_stack((b, np.ones(len(b))))
    F = _fundamental(params)
    la = (F @ aa.T).T
    lb = (F.T @ bb.T).T
    numer = np.sum(bb * la, axis=1) ** 2
    denom = np.sum(la[:, :2] ** 2, axis=1) + np.sum(lb[:, :2] ** 2, axis=1)
    if not np.isfinite(numer).all() or not np.isfinite(denom).all():
        raise ValueError("Invalid epipolar geometry")
    return np.sqrt(numer / np.maximum(denom, 1e-30))


def _corners(archive: ZipFile, member: str):
    import cv2
    data = archive.read(member)
    gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    # Calibration JPGs are landscape, whereas source K uses portrait axes.
    # Establish rotation on source FIT frames, then keep it fixed for holdout.
    gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    small = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    ok, candidate = cv2.findChessboardCorners(
        small, (8, 6),
        flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not ok:
        return None
    refined = cv2.cornerSubPix(
        gray, candidate * 2, (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
    )
    return refined.reshape(-1, 2)


def audit_stereo_pair(raw_root: str | Path, external_zip: str | Path,
                      subject="S1", paired_camera="2") -> dict:
    """Fit-frame orientation selection, then non-fit-frame epipolar evaluation."""
    raw_root = Path(raw_root)
    with (raw_root / "calibration/external_calibration.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = [r for r in csv.DictReader(file)
                if r["subject"] == subject and r["cam"] == str(paired_camera)]
    if len(rows) != 1:
        raise ValueError("Expected exactly one calibration row for paired camera")
    row = rows[0]
    main, pair = int(row["main"]), int(row["pair"])
    if pair != int(paired_camera) or main == pair:
        raise ValueError("Expected non-origin paired camera record")
    used_a, used_b = _frame_ids(row["valid_frames_A"]), _frame_ids(row["valid_frames_B"])
    if used_a != used_b:
        raise ValueError("Source selected-frame lists disagree across camera pair")
    names = f"G{main}{pair}"
    base_a = f"C{main}{subject}_{names}"
    base_b = f"C{pair}{subject}_{names}"
    rel = Path(row["path"].replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:3] != (
        "calibration", "external", "matrix"
    ):
        raise ValueError("Unsafe calibration matrix path")
    with np.load(raw_root / rel, allow_pickle=False) as z:
        params = {k: z[k] for k in
                  ("R", "T", "mtx_A", "mtx_B", "dist_A", "dist_B")}
    fit = sorted(used_a)
    holdout = [i for i in range(1, 61) if i not in used_a]
    scores = {"fit": [], "holdout": []}
    with ZipFile(external_zip) as archive:
        members = set(archive.namelist())
        for split, ids in (("fit", fit), ("holdout", holdout)):
            for i in ids:
                name_a = f"{base_a}/{base_a}_{i:04d}.jpg"
                name_b = f"{base_b}/{base_b}_{i:04d}.jpg"
                if name_a not in members or name_b not in members:
                    continue
                a, b = _corners(archive, name_a), _corners(archive, name_b)
                if a is not None and b is not None:
                    scores[split].append((i, a, b))
    if not scores["fit"]:
        raise ValueError("No source fit frames have detectable paired chessboards")
    choices = ("same", "flip_horizontal", "flip_vertical", "rotate_180")
    fit_medians = {}
    for orientation in choices:
        errors = [np.median(sampson_distance_pixels(a, _symmetry(b, orientation), params))
                  for _, a, b in scores["fit"]]
        fit_medians[orientation] = float(np.median(errors))
    chosen = min(choices, key=lambda k: fit_medians[k])
    held = [float(np.median(sampson_distance_pixels(a, _symmetry(b, chosen), params)))
            for _, a, b in scores["holdout"]]
    train = fit_medians[chosen]
    return {
        "subject": subject, "origin_camera": f"C{main}", "paired_camera": f"C{pair}",
        "source_group": names, "checkerboard_inner_corners": [8, 6],
        "raw_calibration_jpeg_rotation": "90_degrees_counterclockwise",
        "fit_frames_used_to_choose_corner_orientation": len(scores["fit"]),
        "source_fit_indices_count": len(fit),
        "holdout_indices_declared": len(holdout),
        "holdout_detectable_pairs": len(held),
        "holdout_missing_or_undetected_pairs": len(holdout) - len(held),
        "training_selected_symmetry": chosen,
        "fit_median_sampson_epipolar_px": round(train, 4),
        "holdout_median_pair_sampson_epipolar_px": (
            round(float(np.median(held)), 4) if held else None),
        "holdout_p90_pair_sampson_epipolar_px": (
            round(float(np.percentile(held, 90)), 4) if held else None),
        "holdout_pairs_median_epipolar_below_2px": sum(x <= 2 for x in held),
        "uses_original_source_RT_and_intrinsics_unchanged": True,
        "fit_frame_indices_excluded_from_holdout": True,
        "heldout_epipolar_geometry_evaluated": bool(held),
        "independent_3d_reprojection_accuracy_verified": False,
        "camera_vicon_physical_sync_verified": False,
        "notes": ("Source fit frames choose checkerboard 180/flip orientation only. "
                  "Sampson epipolar distance on non-fit JPG pairs is not independent "
                  "3D Vicon calibration or exposure-time synchronization. "
                  "Detection failures remain excluded and counted."),
    }
