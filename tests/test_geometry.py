import math

from sumedipose.geometry import angle_from_markers, joint_angle, knee_angles


def test_right_angle():
    assert math.isclose(joint_angle([1, 0, 0], [0, 0, 0], [0, 1, 0]), 90.0)


def test_straight_angle():
    assert math.isclose(joint_angle([1, 0, 0], [0, 0, 0], [-1, 0, 0]), 180.0)


def test_zero_length_is_nan():
    assert math.isnan(joint_angle([0, 0, 0], [0, 0, 0], [1, 0, 0]))


def test_named_joints():
    r = {
        "point_ids": ["LHJC", "LKJC", "LAJC", "RHJC", "RKJC", "RAJC"],
        "xyz": [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [-1, 0, 0]],
    }
    assert angle_from_markers(r, "LHJC", "LKJC", "LAJC") == 90.0
    assert knee_angles(r) == {"left_knee_degrees": 90.0, "right_knee_degrees": 180.0}
