"""Synthetic-only parity tests. No RGB, Vicon raw data or real participants."""
import numpy as np
import pytest

from sumedipose.mediapipe_parity import (
    aggregate, compare_holdout, fit_parity_similarity,
)
from sumedipose.position_holdout import ANCHORS, CAMERAS, VARIANTS, root_relative


def _points(n=16):
    base=np.array([
        [-.10,.03,.10],[-.12,-.42,.08],[-.08,-.85,.04],
        [.10,-.03,-.10],[.14,-.45,-.12],[.11,-.88,-.06],
    ])
    frames=[]
    for i in range(n):
        x=base.copy()
        x[:,2]+=i*.007
        x[1,0]-=i*.002
        x[4,0]+=i*.003
        frames.append(root_relative(x))
    return np.stack(frames)


def _triples(x,y,offset=0):
    return [(i+offset,x[i],y[i]) for i in range(len(x))]


def test_reflected_model_frame_better_than_proper_on_holdout():
    x=_points(16)
    y=x@np.diag([-1.,1.,1.])*1000
    # Model world has mirrored handedness compared to the Vicon reference.
    fits=compare_holdout(_triples(x[:8],y[:8]),
                         _triples(x[8:14],y[8:14],offset=8),
                         _triples(x[10:16],y[10:16],offset=10))
    assert fits["proper"]["determinant"]==1
    assert fits["reflected"]["determinant"]==-1
    assert fits["reflected"]["A3_heldout"][
        "mean_discrepancy_pct_reference_femur_length"] == 0
    assert fits["reflected"]["A4_cross_action"][
        "mean_discrepancy_pct_reference_femur_length"] == 0
    assert fits["proper"]["A4_cross_action"][
        "mean_discrepancy_pct_reference_femur_length"] > 1


def test_proper_model_frame_better_than_reflected_on_holdout():
    x=_points(16)
    y=x@np.diag([1.,-1.,-1.])*1000
    fit=compare_holdout(_triples(x[:8],y[:8]),
                        _triples(x[8:14],y[8:14],offset=8),
                        _triples(x[10:16],y[10:16],offset=10))
    assert fit["proper"]["A4_cross_action"][
        "mean_discrepancy_pct_reference_femur_length"] == 0
    assert fit["reflected"]["A4_cross_action"][
        "mean_discrepancy_pct_reference_femur_length"] > 1


def test_parity_fit_is_orthogonal_positive_scale_and_explicit():
    x=_points(10)
    reflected=x@np.diag([-1.,1.,1.]) * 900
    mat,scale=fit_parity_similarity(x[:8],reflected[:8],determinant=-1)
    assert np.linalg.det(mat) == pytest.approx(-1,abs=1e-8)
    assert scale == pytest.approx(900,rel=1e-7)
    with pytest.raises(ValueError,match="determinant"):
        fit_parity_similarity(x[:8],reflected[:8],determinant=0)
    with pytest.raises(ValueError,match=">=6"):
        fit_parity_similarity(x[:5],reflected[:5],determinant=-1)
    with pytest.raises(ValueError,match="Degenerate"):
        fit_parity_similarity(np.zeros((8,6,3)),np.zeros((8,6,3)),determinant=-1)


def _row(subject,camera,model,proper_A3=30.,reflect_A3=20.,
         proper_A4=32.,reflect_A4=21.):
    def check(a3,a4,det):
        return {
            "determinant": det,
            "A3_heldout": {
                "mean_discrepancy_pct_reference_femur_length": a3,
                "valid_evaluation_frames": 11,
            },
            "A4_cross_action": {
                "mean_discrepancy_pct_reference_femur_length": a4,
                "valid_evaluation_frames": 22,
            },
        }
    return {
        "subject":subject,"camera":camera,"model":model,
        "A3_source_audit":{"model_images_attempted":24,"pose_detections":23},
        "A4_source_audit":{"model_images_attempted":24,"pose_detections":20},
        "fits":{"proper":check(proper_A3,proper_A4,+1),
                "reflected":check(reflect_A3,reflect_A4,-1)},
    }


def test_five_calibration_groups_are_never_inflated():
    rows=[_row(s,c,m) for s in ANCHORS for c in CAMERAS for m in VARIANTS]
    d=aggregate(rows)
    assert d["model_report_pairs"]==20
    assert d["model_image_inferences"]==960
    assert d["pose_detections"]==860
    assert len(d["cells"])==8
    assert d["paired_parity_preference_stable_A3_to_A4_count"]==20
    assert d["independent_calibration_groups"]==5
    assert d["new_independent_calibration_groups"]==0
    assert d["framewise_Vicon_GT_hip_centre_used_to_root_relative"] is True
    assert d["absolute_MPJPE_mm_validated"] is False
    assert all(c["reflected_better_group_count"]==5 for c in d["cells"])
    assert "subject" not in d


def test_missing_anchor_or_camera_rejected():
    rows=[_row(s,c,m) for s in ANCHORS for c in CAMERAS for m in VARIANTS]
    with pytest.raises(ValueError):
        aggregate(rows[:-1])
    rows[-1]=_row("S1","C1","lite")
    with pytest.raises(ValueError,match="Missing/duplicate"):
        aggregate(rows)
