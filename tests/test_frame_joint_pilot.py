"""Pure synthetic tests; no SUMediPose participant data required."""
import numpy as np
import pytest

from sumedipose.frame_joint_pilot import (
    SHIFTS, aggregate_pilot, reference_angles, shift_sensitivity,
)


def test_shift_zero_best_for_exact_reference():
    refs = {"left": np.arange(100, dtype=float)}
    rows = [(i, "left", float(i)) for i in (20, 30, 50, 70)]
    got = shift_sensitivity(rows, refs)
    assert got["baseline_valid"] == 4
    assert got["matched_valid"] == 4
    assert got["baseline_mae_deg"] == 0
    assert got["best_exploratory_index_shift"] == 0
    assert got["shift_zero_is_best_or_tied"]
    assert not got["index_shift_is_time_sync_proof"]


def test_preselected_record_shift_is_diagnostic_not_hardware_timing():
    refs = {"left": np.arange(100, dtype=float)}
    rows = [(i, "left", float(i + 3)) for i in (20, 30, 50, 70)]
    got = shift_sensitivity(rows, refs)
    assert got["baseline_mae_deg"] == 3
    assert got["best_exploratory_index_shift"] == 3
    assert got["best_exploratory_mae_deg"] == 0
    assert got["best_vs_zero_in_sample_mae_delta_deg"] == 3
    assert not got["index_shift_is_time_sync_proof"]


def test_common_sample_mask_excludes_edges_without_changing_baseline():
    refs = {"left": np.arange(100, dtype=float)}
    rows = [(0, "left", 0.0), (50, "left", 50.0), (99, "left", 99.0)]
    got = shift_sensitivity(rows, refs)
    assert got["baseline_valid"] == 3
    assert got["matched_valid"] == 1
    assert got["same_observation_subset_at_every_shift"]


def test_no_usable_observations_and_invalid_shifts():
    refs = {"left": np.asarray([float("nan")] * 100)}
    got = shift_sensitivity([(50, "left", 30.)], refs)
    assert got["baseline_valid"] == 0
    assert got["matched_valid"] == 0
    with pytest.raises(ValueError):
        shift_sensitivity([], refs, shifts=(-1, 1))


def test_references_and_angle_geometry():
    def record(prox, joint, dist):
        return {
            "point_ids": ["LHJC", "LKJC", "LAJC", "RHJC", "RKJC", "RAJC"],
            "xyz": [prox, joint, dist, prox, joint, dist],
        }
    records = [record([0, 1, 0], [0, 0, 0], [0, -1, 0])]
    result = reference_angles(records, "xyz")
    assert result["left"][0] == pytest.approx(180)
    assert result["right"][0] == pytest.approx(180)
