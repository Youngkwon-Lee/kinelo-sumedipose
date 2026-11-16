"""Stereo frame-offset sensitivity audit using source calibration checkerboard JPGs.

Lag is a file-number offset, not a measured exposure-clock delta.
"""
from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .heldout_calibration import (
    _frame_ids, _corners, _symmetry, sampson_distance_pixels, audit_stereo_pair,
)

from .board3d import triangulated_board_metrics, aggregate_board_metrics
from .board_scale import published_board_scale_evidence


def candidate_lag_indices(fit_ids, detected_a, detected_b, lag):
    """Return A-frame IDs for valid, non-fit A[i]/B[i+lag] stereo pairs."""
    if not isinstance(lag, int):
        raise ValueError("lag must be integer")
    usable_a = set(detected_a) - set(fit_ids)
    usable_b = set(detected_b) - set(fit_ids)
    return sorted(i for i in usable_a if i + lag in usable_b)


def lag_summary(raw_root, external_zip, subject="S1",
                paired_camera="5", max_lag=5):
    """Aggregate correspondence residuals over original, held-out board JPGs.

    No raw images or per-frame detected coordinates appear in the result.
    Orientation and source R/T/K are fixed without using held-out images.
    """
    if not 0 <= max_lag <= 15:
        raise ValueError("max_lag must be between 0 and 15")
    raw_root = Path(raw_root)
    paired_camera = str(paired_camera).removeprefix("C")
    baseline = audit_stereo_pair(raw_root, external_zip, subject, paired_camera)
    with (raw_root / "calibration/external_calibration.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = [row for row in csv.DictReader(file)
                if row["subject"] == subject
                and row["cam"] == paired_camera]
    if len(rows) != 1:
        raise ValueError("Exactly one source calibration row required")
    row = rows[0]
    fit_a, fit_b = _frame_ids(row["valid_frames_A"]), _frame_ids(row["valid_frames_B"])
    if fit_a != fit_b:
        raise ValueError("Source fit frame IDs disagree")
    rel = Path(row["path"].replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:3] != (
        "calibration", "external", "matrix"
    ):
        raise ValueError("Unsafe source calibration matrix path")
    with np.load(raw_root / rel, allow_pickle=False) as z:
        params = {k: z[k] for k in ("R", "T", "mtx_A", "mtx_B", "dist_A", "dist_B")}

    group = baseline["source_group"]
    base_a = f"C{row['main']}{subject}_{group}"
    base_b = f"C{row['pair']}{subject}_{group}"
    detected = {"A": {}, "B": {}}
    with ZipFile(external_zip) as z:
        names = set(z.namelist())
        for i in range(1, 61):
            for key, base in (("A", base_a), ("B", base_b)):
                member = f"{base}/{base}_{i:04d}.jpg"
                if member in names:
                    point = _corners(z, member)
                    if point is not None:
                        detected[key][i] = point
    symmetry = baseline["training_selected_symmetry"]
    a, b = detected["A"], detected["B"]
    def score(i, j):
        return float(np.median(sampson_distance_pixels(
            a[i], _symmetry(b[j], symmetry), params
        )))

    # Adjacent board point motion is a coarse proxy for lag discriminability.
    motion = []
    for i in sorted(b):
        if i + 1 in b and i not in fit_a and i + 1 not in fit_a:
            motion.append(float(np.median(np.linalg.norm(
                _symmetry(b[i+1], symmetry) - _symmetry(b[i], symmetry),
                axis=1,
            ))))

    lags = []
    zero_idx = set(candidate_lag_indices(fit_a, a, b, 0))
    zero_scores = {i: score(i, i) for i in zero_idx}
    for lag in range(-max_lag, max_lag + 1):
        idx = candidate_lag_indices(fit_a, a, b, lag)
        values = [score(i, i + lag) for i in idx]
        common = sorted(set(idx) & zero_idx)
        # Matched-anchor comparison controls changing detection coverage per lag.
        common_shifted = [score(i, i + lag) for i in common]
        common_zero = [zero_scores[i] for i in common]
        lags.append({
            "offset_file_frame_numbers": lag,
            "evaluated_pairs": len(values),
            "median_frame_sampson_epipolar_px": (
                round(float(np.median(values)), 4) if values else None),
            "p90_frame_sampson_epipolar_px": (
                round(float(np.percentile(values, 90)), 4) if values else None),
            "common_zero_anchors": len(common),
            "matched_zero_median_px": (
                round(float(np.median(common_zero)), 4) if common else None),
            "matched_shifted_median_px": (
                round(float(np.median(common_shifted)), 4) if common else None),
            "matched_shift_minus_zero_median_px": (
                round(float(np.median(common_shifted) - np.median(common_zero)), 4)
                if common else None),
        })
    ranked = [x for x in lags if x["evaluated_pairs"] >= 10]
    best = min(ranked, key=lambda x:x["median_frame_sampson_epipolar_px"]) if ranked else None
    boards = [
        triangulated_board_metrics(a[i], _symmetry(b[i], symmetry), params)
        for i in sorted(zero_idx)
    ]
    board_summary = aggregate_board_metrics(boards)
    return {
        "session":subject,
        "group":group,
        "origin_camera":baseline["origin_camera"],
        "paired_camera":baseline["paired_camera"],
        "source_fit_indices_excluded":len(fit_a),
        "stereo_board_orientation_selected_on_fit_only":symmetry,
        "source_intrinsic_extrinsic_parameters_frozen":True,
        "camera_pair_image_indices_available":[len(a),len(b)],
        "median_adjacent_board_motion_px":round(float(np.median(motion)),4) if motion else None,
        "lags":lags,
        "lowest_median_error_offset_observed":best["offset_file_frame_numbers"] if best else None,
        "zero_index_holdout_board_geometry": board_summary,
        "source_checkerboard_world_scaling":1,
        "physical_square_size_mm_known":True,
        "published_nominal_board_scale":published_board_scale_evidence(),
        "measured_actual_board_square_size_mm_verified":False,
        "zero_index_stereo_error_px": next(
            x["median_frame_sampson_epipolar_px"] for x in lags
            if x["offset_file_frame_numbers"] == 0),
        "is_zero_index_minimum_across_tested_offsets": bool(
            best is not None and best["offset_file_frame_numbers"]==0),
        "physical_exposure_sync_established":False,
        "file_index_lag_is_real_time_offset":False,
        "interpretation_note":(
            "A minimum at any shifted calibration-image index cannot establish "
            "a true exposure-time offset. Different anchor availability, board "
            "movement, detection error and intrinsic/extrinsic calibration error "
            "can change the epipolar metric; use matched-zero comparisons."
        ),
    }
