"""Only generated synthetic point correspondences; no original calibration images."""
import math

import numpy as np
import pytest

from sumedipose.heldout_calibration import _frame_ids, _symmetry, sampson_distance_pixels


def test_training_frame_paths_are_identified_exactly():
    value = "['calibration/external/frames/S1/C1S1_G12/C1S1_G12_0013.jpg', 'C1S1_G12_0020.jpg']"
    assert _frame_ids(value) == {13, 20}
    with pytest.raises(ValueError):
        _frame_ids("['other-file.png']")
    with pytest.raises((ValueError, SyntaxError)):
        _frame_ids("not-a-list")


def test_grid_reorientation_is_explicit():
    grid = np.arange(6 * 8 * 2).reshape(-1, 2).astype(float)
    assert np.all(_symmetry(grid, "same") == grid)
    assert np.all(_symmetry(grid, "rotate_180") == grid[::-1])
    assert np.array_equal(_symmetry(grid, "flip_vertical").reshape(6,8,2)[0], grid.reshape(6,8,2)[-1])
    with pytest.raises(ValueError):
        _symmetry(grid, "unsupported")


def test_correct_stereo_geometry_has_near_zero_sampson_error():
    # Source-style K/R/T, synthesize corresponding 3D rays for two cameras.
    K1 = np.array([[900.,0.,575.],[0.,910.,900.],[0.,0.,1.]])
    K2 = np.array([[905.,0.,577.],[0.,915.,899.],[0.,0.,1.]])
    a = .07
    R = np.array([[math.cos(a),0,math.sin(a)],[0,1,0],
                  [-math.sin(a),0,math.cos(a)]])
    T = np.array([[0.25],[0.01],[-0.03]])
    X = np.stack((np.linspace(-.4,.4,48), np.linspace(-.3,.3,48),
                  np.linspace(3.,5.,48)),axis=1)
    Xb = (R @ X.T + T).T
    def project(points, K):
        uv = (K @ points.T).T
        return uv[:, :2] / uv[:, 2, None]
    p1, p2 = project(X,K1), project(Xb,K2)
    params = {"mtx_A":K1,"mtx_B":K2,"R":R,"T":T,
              "dist_A":np.zeros((1,5)),"dist_B":np.zeros((1,5))}
    errors = sampson_distance_pixels(p1,p2,params)
    assert float(np.max(errors)) < 1e-6
    wrong = p2 + np.array([0., 12.])
    assert float(np.median(sampson_distance_pixels(p1, wrong, params))) > 1
    with pytest.raises(ValueError):
        sampson_distance_pixels(p1,p2[:-1],params)


def test_cli_holdout_requires_external_zip():
    from sumedipose.cli import main
    with pytest.raises(SystemExit) as result:
        main(["--root", ".", "--stereo-holdout"])
    assert result.value.code == 2
