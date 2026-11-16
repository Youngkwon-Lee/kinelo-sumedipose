"""Synthetic-only tests; no human or original checkerboard images required."""
import math

import numpy as np
import pytest

from sumedipose.calibration_uncertainty import (
    _summarize, _raw_sampson, relative_pose_disagreement,
)


def rotation_y(degrees):
    t = np.radians(degrees)
    return np.array([[np.cos(t), 0., np.sin(t)],
                     [0., 1., 0.],
                     [-np.sin(t), 0., np.cos(t)]])


def test_relative_pose_disagreement_zero_for_consistent_source():
    ra = rotation_y(15)
    ta = np.array([[3.], [1.], [80.]])
    source_r = rotation_y(-7)
    source_t = np.array([[30.], [.5], [5.]])
    rb = source_r @ ra
    tb = source_r @ ta + source_t
    rotation, translation = relative_pose_disagreement(
        ra, ta, rb, tb, source_r, source_t
    )
    assert rotation < 1e-5
    assert translation < 1e-9


def test_relative_pose_disagreement_detects_source_rotation_and_translation():
    ra = np.eye(3)
    ta = np.array([[0.], [0.], [80.]])
    source_r = rotation_y(1)
    source_t = np.array([[30.], [0.], [5.]])
    actual_r = rotation_y(5)
    actual_t = source_t + np.array([[1.5], [0.], [-.5]])
    rb, tb = actual_r, actual_r @ ta + actual_t
    rotation, translation = relative_pose_disagreement(
        ra, ta, rb, tb, source_r, source_t
    )
    assert 3.999 < rotation < 4.001
    assert abs(translation - math.hypot(1.5, .5)) < 1e-9


def test_summary_empty_and_nonfinite_values():
    assert _summarize([]) == {"n":0,"median":None,"p90":None}
    v = _summarize([0., 1., 2., 3.])
    assert v["n"] == 4
    assert v["median"] == 1.5
    with pytest.raises(ValueError):
        _summarize([float("nan")])


def test_raw_epipolar_residual_zero_for_perfect_synthetic_stereo():
    k1 = np.array([[1200.,0.,575.],[0.,1220.,900.],[0.,0.,1.]])
    k2 = np.array([[1205.,0.,571.],[0.,1210.,894.],[0.,0.,1.]])
    R = rotation_y(3)
    T = np.array([[25.], [.6], [3.]])
    X = np.column_stack((
        np.linspace(-.5, .5, 48),
        np.linspace(-.4, .4, 48),
        np.linspace(70., 85., 48),
    ))
    def project(points, K):
        p = (K @ points.T).T
        return p[:, :2] / p[:, 2, None]
    a = project(X, k1)
    b = project((R @ X.T + T).T, k2)
    params = {"R": R, "T": T, "mtx_A": k1, "mtx_B": k2}
    assert np.max(_raw_sampson(a, b, params)) < 1e-6
    b[:, 1] += 14.
    assert np.median(_raw_sampson(a, b, params)) > 1.


def test_calibration_uncertainty_cli_requires_verified_original_archive():
    from sumedipose.cli import main
    with pytest.raises(SystemExit) as e:
        main(["--root", ".", "--calibration-uncertainty"])
    assert e.value.code == 2
