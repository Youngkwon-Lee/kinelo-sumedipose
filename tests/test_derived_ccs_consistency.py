"""Synthetic-only Vicon-derived WCS/CCS chirality and scale tests."""
from copy import deepcopy

import numpy as np
import pytest

from sumedipose.derived_ccs_consistency import (
    aggregate, orthogonal_similarity, paired_error,
)


def _points(n=24):
    base = np.asarray([
        [-90., 600., 50.], [-135., 210., 62.], [-110., -160., 30.],
        [90., 590., 55.], [135., 195., 66.], [108., -185., 42.],
    ])
    seq = []
    for i in range(n):
        coords = base.copy()
        coords[:, 2] += (i - 12) * (coords[:, 1] / 900)
        coords[1:3, 0] += i * 2.3
        coords[4:, 0] -= i * 1.6
        seq.append(coords)
    return np.stack(seq)


def test_reflection_and_sixty_fold_scale_remain_fixed_on_holdout():
    x = _points()
    reflection = np.diag([-1., 1., 1.])
    y = x @ reflection / 60.0 + [12., -9., 3.]
    good = orthogonal_similarity(x[:12], y[:12], determinant=-1)
    bad = orthogonal_similarity(x[:12], y[:12], determinant=1)
    assert np.linalg.det(good[0]) == pytest.approx(-1, abs=1e-9)
    assert np.linalg.det(bad[0]) == pytest.approx(1, abs=1e-9)
    assert good[1] == pytest.approx(1/60, rel=1e-12)
    heldout = paired_error(x[12:], y[12:], good)
    wrong = paired_error(x[12:], y[12:], bad)
    assert heldout["mean_discrepancy_WCS_native_units"] == 0
    assert heldout["mean_discrepancy_pct_reference_femur"] == 0
    assert wrong["mean_discrepancy_pct_reference_femur"] > 1
    assert heldout["evaluated_frames"] == 12


def test_proper_rigid_rotation_is_not_falsely_forced_to_reflect():
    x = _points()
    y = x * .02 + [1., 4., 2.]
    fit = orthogonal_similarity(x[:12], y[:12], determinant=1)
    assert fit[1] == pytest.approx(.02, abs=1e-12)
    assert np.linalg.det(fit[0]) == pytest.approx(1, abs=1e-9)
    assert paired_error(x[12:], y[12:], fit)["mean_discrepancy_WCS_native_units"] == 0


def test_invalid_and_degenerate_geometry_fails_closed():
    with pytest.raises(ValueError, match="Determinant"):
        orthogonal_similarity(_points()[:12], _points()[:12], determinant=0)
    with pytest.raises(ValueError, match="at least six"):
        orthogonal_similarity(_points()[:5], _points()[:5], determinant=-1)
    z = np.zeros((12, 6, 3))
    with pytest.raises(ValueError, match="Degenerate"):
        orthogonal_similarity(z, z, determinant=1)
    with pytest.raises(ValueError, match="Nonfinite"):
        x = _points().copy()
        x[0, 0, 0] = np.nan
        orthogonal_similarity(x, x, determinant=-1)


def _fake_rows():
    rows = []
    sizes = (4, 4, 6, 6, 6)
    roots = ("S1", "S6", "S11", "S17", "S23")
    participants = [f"S{i}" for i in range(1, 29) if i not in (5, 10)]
    mapping = {}
    for root, group in zip(roots, (
        (1, 2, 3, 4), (6, 7, 8, 9),
        (11, 12, 13, 14, 15, 16),
        (17, 18, 19, 20, 21, 22),
        (23, 24, 25, 26, 27, 28),
    )):
        mapping.update({f"S{i}": root for i in group})
    for p in participants:
        for cam in ("C1", "C6"):
            better = {
                "fit_scale_CCS_units_per_WCS_native": 1/60,
                "A3_heldout": {"mean_discrepancy_WCS_native_units": .5,
                               "mean_discrepancy_pct_reference_femur": .15},
            }
            worse = {
                "fit_scale_CCS_units_per_WCS_native": .014,
                "A3_heldout": {"mean_discrepancy_WCS_native_units": 160.,
                               "mean_discrepancy_pct_reference_femur": 40.},
            }
            if p in roots:
                for item, raw, pct in ((better, .7, .18), (worse, 200., 50.)):
                    item["A4_transfer"] = {
                        "mean_discrepancy_WCS_native_units": raw,
                        "mean_discrepancy_pct_reference_femur": pct,
                    }
            rows.append({
                "subject": p, "camera": cam,
                "A3": {
                    "source_wcs_to_ccs_segment_length_ratio_median": 60.,
                    "original_WCS_timestamp_keys": [],
                    "original_camera_timestamp_keys": [],
                },
                "A4": {"original_WCS_timestamp_keys": []} if p in roots else None,
                "comparison": {
                    "proper_rotation": worse,
                    "axis_reversal_allowed": better,
                },
            })
    return rows, mapping


def test_grouped_26_subjects_do_not_inflate_independent_calibration_n():
    rows, mapping = _fake_rows()
    d = aggregate(rows, mapping)
    assert d["status"].startswith("complete_derived_Vicon")
    assert d["A3_view_recordings"] == 52
    assert d["A4_view_recordings"] == 10
    assert len(d["group_results"]) == 4
    assert d["WCS_to_derived_CCS_segment_length_ratio_median"] == 60
    assert d["A3_WCS_records_without_timestamp_keys"] == 52
    assert all(z["reflected_fit_better_in_subject_camera_cases"] == z[
        "source_subject_records"] for z in d["group_results"])
    assert d["independent_calibration_groups"] == 5
    assert d["new_independent_calibration_groups"] == 0
    assert d["physical_mpjpe_mm_validated"] is False
    assert d["no_physical_camera_Vicon_common_clock_evidence"] is True
    assert "rows" not in d and "subject" not in d and "subject_map" not in d


def test_incomplete_and_repeated_participants_fail_before_aggregation():
    rows, mapping = _fake_rows()
    with pytest.raises(ValueError, match="Expected original 26"):
        aggregate(rows[:-1], mapping)
    rows[1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="Expected original 26"):
        aggregate(rows, mapping)
