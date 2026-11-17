"""Synthetic safeguards for A3 -> withheld A3/A4 constant-bias transfer."""
import numpy as np
import pytest

from sumedipose.knee_bias_transfer import (
    ANCHORS, BANDS, CAMERAS, CORRECTIONS, SPLITS, VARIANTS,
    _band, angle_observations, evaluation_metrics, invariant_under_axis_flip,
    make_summary, train_constant_bias,
)


def _frame(slot, pred_knee=160., ref_knee=140.):
    # Independent XY vectors, signed included angle around knee origin.
    def pts(degrees):
        radians = np.deg2rad(degrees)
        return np.asarray([
            [1., 0., 0.], [0., 0., 0.],
            [np.cos(radians), np.sin(radians), 0.],
        ])
    a = pts(pred_knee)
    b = pts(ref_knee)
    return slot, np.vstack((a, a + [0., 0., 2.])), np.vstack((b, b + [0., 0., 2.]))


def test_orthogonal_axis_mirror_and_rotation_cannot_fix_angle():
    frame = _frame(0, 168, 140)
    assert invariant_under_axis_flip(frame)
    obs = angle_observations([frame])
    assert len(obs) == 2
    assert obs[0]["model"] == pytest.approx(168)
    assert obs[0]["vicon"] == pytest.approx(140)


def test_train_a3_constant_offset_never_uses_heldout_labels():
    training = angle_observations([_frame(i,160,140) for i in range(8)])
    bias = train_constant_bias(training)
    assert bias == pytest.approx(20)
    tests = angle_observations([_frame(13,175,145),_frame(14,150,110)])
    metrics = evaluation_metrics(tests,bias)
    assert metrics["valid_knee_angle_samples"] == 4
    assert metrics["baseline_mae_deg"] > metrics["corrected_mae_deg"]
    # No clipping at 0/180 is allowed
    assert evaluation_metrics(tests,-100)["corrected_angle_out_of_0_180_count"] == 4


def test_training_fails_if_fewer_than_six_valid_frames():
    with pytest.raises(ValueError,match="Insufficient"):
        train_constant_bias(angle_observations([_frame(i) for i in range(5)]))


def test_depth_invariant_with_reference_phase_bands():
    assert _band(119.9)==BANDS[0]
    assert _band(120)==BANDS[1]
    assert _band(150)==BANDS[2]


def _dummy(sub,c,m,bias,phase_gt=100.):
    # Five distinct group biases; peer LOCO must exclude target
    return {
        "subject":sub,"camera":c,"model":m,
        "a3_train_mean_signed_bias_deg":bias,
        "a3_train_valid_frames":12,
        "a3_holdout_valid_frames":12,
        "a4_transfer_valid_frames":12,
        "a3_images_attempted":24,
        "a4_images_attempted":24,
        "a3_pose_detections":24,
        "a4_pose_detections":24,
        "_A3_heldout":[{"slot":i,"model":phase_gt+bias+1,"vicon":phase_gt,
                        "band":_band(phase_gt)} for i in range(12) for _ in (0,1)],
        "_A4_transfer":[{"slot":i,"model":phase_gt+bias+2,"vicon":phase_gt,
                        "band":_band(phase_gt)} for i in range(12) for _ in (0,1)],
    }


def test_loo_group_calibration_never_uses_target_A3_training_bias():
    rows = [_dummy(s,c,m,float(i+1)*10) for i,s in enumerate(ANCHORS)
            for c in CAMERAS for m in VARIANTS]
    overview,details = make_summary(rows)
    assert overview["anchor_groups"]==5
    assert overview["total_model_recording_pairs"]==40
    assert overview["total_image_inferences"]==960
    local = next(x for x in details if x["subject"]=="S1" and
                 x["camera"]=="C6" and x["model"]=="full" and
                 x["test_split"]=="A4_transfer" and
                 x["calibration_method"]==CORRECTIONS[0])
    loco = next(x for x in details if x["subject"]=="S1" and
                 x["camera"]=="C6" and x["model"]=="full" and
                 x["test_split"]=="A4_transfer" and
                 x["calibration_method"]==CORRECTIONS[1])
    assert local["fitted_A3_train_bias_deg"]==pytest.approx(10)
    assert loco["fitted_A3_train_bias_deg"]==pytest.approx(35)
    assert local["corrected_mae_deg"]==pytest.approx(2)
    assert loco["corrected_mae_deg"]==pytest.approx(23)
    assert len(overview["results"])==16


def test_missing_or_duplicate_anchor_fails_closed():
    rows = [_dummy(s,c,m,10.) for s in ANCHORS for c in CAMERAS for m in VARIANTS]
    with pytest.raises(ValueError,match="Expected"):
        make_summary(rows[1:])
    with pytest.raises(ValueError,match="Expected"):
        make_summary(rows[:-1]+[rows[0]])


def test_empty_phase_report_is_explicitly_zero_not_filled():
    rows = [_dummy(s,c,m,10.) for s in ANCHORS for c in CAMERAS for m in VARIANTS]
    aggregate,_=make_summary(rows)
    row=next(x for x in aggregate["results"] if x["camera"]=="C6" and
             x["model"]=="lite" and x["test_split"]=="A4_transfer" and
             x["method"]==CORRECTIONS[1])
    assert row["by_vicon_angle"]["deep_lt120"]["groups_with_data"]==5
    assert row["by_vicon_angle"]["extended_ge150"]["groups_with_data"]==0
