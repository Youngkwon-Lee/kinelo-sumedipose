"""All geometry generated synthetically; no original calibration photos or subjects."""
import math

import cv2
import numpy as np
import pytest

from sumedipose.board3d import triangulated_board_metrics, aggregate_board_metrics
from sumedipose.stereo_lag import candidate_lag_indices


def _synthetic_board():
    grid = np.mgrid[:8, :6].T.reshape(-1, 2).astype(float)
    original = np.column_stack((grid, np.zeros(48)))
    Rboard, _ = cv2.Rodrigues(np.array([.08, -.05, .03]))
    X = (Rboard @ original.T).T + np.array([-3.5, -2.5, 88.])
    K1 = np.array([[1500., 0, 560.], [0, 1510., 900.], [0, 0, 1.]])
    K2 = np.array([[1490., 0, 574.], [0, 1500., 905.], [0, 0, 1.]])
    R, rv = None, None
    R, _ = cv2.Rodrigues(np.array([0.02, -0.08, 0.01]))
    rv, _ = cv2.Rodrigues(R)
    T = np.array([[30.], [.5], [6.]])
    zero = np.zeros((5, 1))
    params = {"R": R, "T": T, "mtx_A": K1, "mtx_B": K2,
              "dist_A": zero, "dist_B": zero}
    a, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K1, zero)
    b, _ = cv2.projectPoints(X, rv, T, K2, zero)
    return a.reshape(-1, 2), b.reshape(-1, 2), params


def test_stereo_file_offset_candidate_pairs_exclude_source_fit():
    source_fit = {2, 5}
    a = {1: None, 2: None, 3: None, 4: None, 5: None}
    b = {1: None, 2: None, 3: None, 4: None, 5: None}
    assert candidate_lag_indices(source_fit, a, b, 0) == [1, 3, 4]
    assert candidate_lag_indices(source_fit, a, b, 1) == [3]
    assert candidate_lag_indices(source_fit, a, b, -1) == [4]
    with pytest.raises(ValueError):
        candidate_lag_indices(source_fit, a, b, 1.1)


def test_noise_free_stereo_triangulation_has_zero_error_and_square_edges():
    a, b, params = _synthetic_board()
    record = triangulated_board_metrics(a, b, params)
    assert record["geometry_valid"] is True
    assert record["valid_cheirality_points"] == 48
    assert record["median_two_view_reprojection_px"] < 1e-6
    assert abs(record["median_board_edge_length_source_square_units"] - 1) < 1e-7
    assert record["median_abs_board_edge_minus_one_square_units"] < 1e-6
    assert record["board_plane_rmse_source_square_units"] < 1e-6


def test_misaligned_synthetic_corners_raise_reprojection_and_square_error():
    a, b, params = _synthetic_board()
    b_noisy = b.copy()
    b_noisy[:, 1] += 5 * np.sin(np.arange(48) * 0.2)
    clean = triangulated_board_metrics(a, b, params)
    noisy = triangulated_board_metrics(a, b_noisy, params)
    assert noisy["geometry_valid"]
    assert noisy["median_two_view_reprojection_px"] > clean["median_two_view_reprojection_px"]
    assert noisy["median_abs_board_edge_minus_one_square_units"] > clean["median_abs_board_edge_minus_one_square_units"]
    with pytest.raises(ValueError):
        triangulated_board_metrics(a[:40], b, params)


def test_summary_aggregate_counts_invalid_without_inventing_precision():
    a, b, params = _synthetic_board()
    good = triangulated_board_metrics(a, b, params)
    overview = aggregate_board_metrics([good, {"geometry_valid":False}])
    assert overview["valid_frames"] == 1
    assert overview["total_frames_submitted"] == 2
    assert overview["invalid_or_low_depth_frames"] == 1
    assert overview["board_square_size_in_mm_verified"] is False
    assert overview["absolute_3d_world_accuracy_verified"] is False
    assert overview["independent_point_validation"] is False


def test_lag_cli_requires_zip():
    from sumedipose.cli import main
    with pytest.raises(SystemExit) as raised:
        main(["--root", ".", "--stereo-lag"])
    assert raised.value.code == 2
