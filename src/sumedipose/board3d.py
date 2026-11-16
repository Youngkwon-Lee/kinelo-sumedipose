"""Checkerboard-square-unit stereo triangulation self-consistency diagnostics.

Original source stereo R/T and intrinsics stay fixed. Triangulation, two-view
reprojection, board lattice and plane residuals are derived from the same two
detected images; they are not independent Vicon-world 3D ground truth.
"""
from __future__ import annotations

import numpy as np


def triangulated_board_metrics(pts_a, pts_b, params):
    """For 8×6 corner grid, measure image/3D grid consistency in source units."""
    import cv2

    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if pts_a.shape != (48, 2) or pts_b.shape != (48, 2):
        raise ValueError("Expected 48 2D checkerboard corners per camera")
    if not np.isfinite(pts_a).all() or not np.isfinite(pts_b).all():
        raise ValueError("Non-finite 2D corners")

    K1, K2 = params["mtx_A"], params["mtx_B"]
    R, T = params["R"], params["T"].reshape(3, 1)
    a = cv2.undistortPoints(
        pts_a.reshape(-1, 1, 2), K1, params["dist_A"]
    ).reshape(-1, 2)
    b = cv2.undistortPoints(
        pts_b.reshape(-1, 1, 2), K2, params["dist_B"]
    ).reshape(-1, 2)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    Xh = cv2.triangulatePoints(P1, P2, a.T, b.T).T
    if np.any(np.abs(Xh[:, 3]) <= 1e-12):
        return None
    X = Xh[:, :3] / Xh[:, 3, None]
    X2 = (R @ X.T + T).T
    if not np.isfinite(X).all():
        return None
    valid_depth = (X[:, 2] > 0) & (X2[:, 2] > 0)
    if valid_depth.sum() < 40:
        return {"valid_cheirality_points":int(valid_depth.sum()),
                "geometry_valid":False}

    view_a, _ = cv2.projectPoints(
        X, np.zeros(3), np.zeros(3), K1, params["dist_A"],
    )
    view_b, _ = cv2.projectPoints(
        X, cv2.Rodrigues(R)[0], T, K2, params["dist_B"],
    )
    err_a = np.linalg.norm(view_a.reshape(-1, 2) - pts_a, axis=1)
    err_b = np.linalg.norm(view_b.reshape(-1, 2) - pts_b, axis=1)
    all_err = np.r_[err_a[valid_depth], err_b[valid_depth]]

    grid = X.reshape(6, 8, 3)
    # These units are 1/8x6 inner-corner checkerboard square as set by the
    # source's world_scaling=1; NOT a verified physical mm measurement.
    edges = np.r_[
        np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2).ravel(),
        np.linalg.norm(grid[1:] - grid[:-1], axis=2).ravel(),
    ]
    center = X - X.mean(axis=0)
    _, _, v = np.linalg.svd(center, full_matrices=False)
    plane = np.abs(center @ v[-1])
    return {
        "geometry_valid":True,
        "valid_cheirality_points":int(valid_depth.sum()),
        "triangulated_3d_points":len(X),
        "median_two_view_reprojection_px":float(np.median(all_err)),
        "rmse_two_view_reprojection_px":float(np.sqrt(np.mean(all_err**2))),
        "median_board_edge_length_source_square_units":float(np.median(edges)),
        "median_abs_board_edge_minus_one_square_units":float(np.median(np.abs(edges - 1.0))),
        "p90_abs_board_edge_minus_one_square_units":float(np.percentile(np.abs(edges - 1),90)),
        "board_plane_rmse_source_square_units":float(np.sqrt(np.mean(plane**2))),
    }


def aggregate_board_metrics(frame_records):
    """Aggregate per-frame geometry internally, returning only summary scalars."""
    usable = [r for r in frame_records if r and r.get("geometry_valid")]
    if not usable:
        return {"valid_frames":0,"metrics":{}}
    names = (
        "median_two_view_reprojection_px",
        "rmse_two_view_reprojection_px",
        "median_board_edge_length_source_square_units",
        "median_abs_board_edge_minus_one_square_units",
        "p90_abs_board_edge_minus_one_square_units",
        "board_plane_rmse_source_square_units",
    )
    return {
        "valid_frames":len(usable),
        "metrics":{
            k:{
                "median_across_frames":round(float(np.median([v[k] for v in usable])),5),
                "p90_across_frames":round(float(np.percentile([v[k] for v in usable],90)),5),
            } for k in names
        },
        "min_cheirality_positive_points":min(v["valid_cheirality_points"] for v in usable),
        "total_frames_submitted":len(frame_records),
        "invalid_or_low_depth_frames":len(frame_records)-len(usable),
        "board_square_size_in_mm_verified":False,
        "absolute_3d_world_accuracy_verified":False,
        "independent_point_validation":False,
        "interpretation_note":(
            "Triangulated corners and reprojections derive from the same two images, "
            "and checkerboard grid is defined in source square-size=1 units. "
            "Without measured square length, values cannot be converted to mm. "
            "This is a heldout self-consistency diagnostic, NOT independent "
            "3D Vicon point accuracy or exposure timestamp synchronization."
        ),
    }
