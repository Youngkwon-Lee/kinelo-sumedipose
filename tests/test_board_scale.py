"""Synthetic/public-source scale checks; no subject records are involved."""
import math
import pytest

from sumedipose.board_scale import (
    BOARD_SIZE_PAPER_URL, PUBLISHED_NOMINAL_SQUARE_MM,
    SOURCE_EXTRINSIC_WORLD_SCALING,
    board_square_units_to_nominal_mm, published_board_scale_evidence,
)


def test_authors_report_nominal_sixty_mm_per_board_square():
    assert PUBLISHED_NOMINAL_SQUARE_MM == 60.0
    assert SOURCE_EXTRINSIC_WORLD_SCALING == 1.0
    assert "PMC12088771" in BOARD_SIZE_PAPER_URL
    assert board_square_units_to_nominal_mm(1.0) == 60.0
    assert board_square_units_to_nominal_mm(-2.5) == -150.0


def test_nominal_unit_does_not_establish_independent_spatial_accuracy():
    evidence = published_board_scale_evidence()
    assert evidence["author_reported_nominal_square_mm"] == 60.0
    assert evidence["equivalent_nominal_mm_per_source_square_unit"] == 60.0
    assert evidence["independent_physical_board_measurement_verified"] is False
    assert evidence["vicon_world_absolute_3d_accuracy_verified"] is False


def test_nonfinite_nominal_conversions_fail():
    for x in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            board_square_units_to_nominal_mm(x)


def test_board_self_consistency_reports_nominal_scale_separately():
    from sumedipose.board3d import aggregate_board_metrics
    from sumedipose.geometry import joint_angle
    assert math.isclose(joint_angle([1,0,0],[0,0,0],[0,1,0]),90)
    m = aggregate_board_metrics([{
        "geometry_valid":True,
        "median_two_view_reprojection_px":0.4,
        "rmse_two_view_reprojection_px":0.7,
        "median_board_edge_length_source_square_units":0.99,
        "median_abs_board_edge_minus_one_square_units":0.01,
        "p90_abs_board_edge_minus_one_square_units":0.03,
        "board_plane_rmse_source_square_units":0.02,
        "valid_cheirality_points":48
    }])
    assert m["published_nominal_board_scale"]["author_reported_nominal_square_mm"] == 60.0
    assert m["board_square_size_in_mm_verified"] is False
    assert m["absolute_3d_world_accuracy_verified"] is False
