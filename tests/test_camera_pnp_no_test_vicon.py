"""Synthetic-only test-time Vicon-free camera PnP and LOCO scale safeguards."""
from pathlib import Path
import csv
import inspect
import numpy as np
import pytest

from sumedipose.camera_pnp_no_test_vicon import (
    calibration_intrinsics, training_group_scale, fixed_scale_excluding_group,
    predict_camera_points, evaluate_prediction, test_recording as evaluate_recording,
    aggregate_fold_rows,
)
from sumedipose.position_holdout import ANCHORS, CAMERAS, VARIANTS


def _geometry():
    return np.array([
        [-.13,.02,.04],[-.19,-.40,.12],[-.08,-.79,.01],
        [.13,-.02,-.04],[.18,-.43,-.11],[.08,-.83,-.02],
    ])


def _K():
    return np.array([[1100.,0.,580.],[0.,1100.,900.],[0.,0.,1.]])


def test_PnP_model_and_image_only_recover_camera_translation_without_gt():
    import cv2
    signature=inspect.signature(predict_camera_points)
    assert set(signature.parameters)=={
        "model_xyz","image_xy","K","distortion","source_units_per_m"
    }
    xyz=_geometry()
    scale=14.
    K=_K()
    true_R=np.eye(3)
    translation=np.array([1.7,2.2,37.])
    true_camera=xyz*scale+translation
    pixel,_=cv2.projectPoints(
        (xyz*scale).astype("float64"),
        np.zeros(3),translation,_K(),np.zeros((5,1)))
    predicted=predict_camera_points(
        xyz,pixel.reshape(-1,2),K,np.zeros((5,1)),scale)
    assert predicted is not None
    assert predicted["test_ground_truth_queried_for_prediction"] is False
    assert np.max(np.linalg.norm(
        predicted["predicted_xyz_camera_source_units"]-true_camera,
        axis=1))<1e-3
    score=evaluate_prediction(predicted,true_camera)
    assert score["Vicon_GT_not_used_in_location_prediction"] is True
    assert score["validated_physical_MPJPE_mm"] is False
    assert score["root_position_discrepancy_pct_reference_femur"]<1e-2


def test_bad_model_only_inputs_and_gt_scoring_are_separate():
    with pytest.raises(ValueError):
        predict_camera_points(np.zeros((6,2)),np.zeros((6,2)),_K(),
                              np.zeros(5),1.)
    with pytest.raises(ValueError):
        predict_camera_points(_geometry(),np.zeros((6,2)),_K(),
                              np.zeros(5),-1.)
    with pytest.raises(ValueError):
        evaluate_prediction(
            {"predicted_xyz_camera_source_units":_geometry(),
             "pelvis_predicted_xyz_camera_source_units":np.zeros(3)},
            np.zeros((3,3)))


def _training(g="S6", val=17.):
    points=_geometry()
    return [
        {"slot":i,
         "model_3d":points,
         "reference_camera_xyz":points*val}
        for i in range(12)
    ]


def test_LOCO_scale_uses_only_other_four_calibration_groups():
    inputs={g:_training(g,16.+index) for index,g in enumerate(ANCHORS)}
    # The heldout group may be radically different and MUST NOT affect its fit.
    train=fixed_scale_excluding_group(inputs,"S1")
    inputs["S1"]=_training(val=1_000_000.)
    assert fixed_scale_excluding_group(inputs,"S1")==pytest.approx(train)
    # Four equal group medians 17,18,19,20 => median 18.5.
    assert train==pytest.approx(18.5)
    with pytest.raises(ValueError,match="Test frame"):
        training_group_scale([dict(x,slot=12) for x in _training()])
    with pytest.raises(ValueError,match="Not enough"):
        training_group_scale(_training()[:5])


def test_source_intrinsics_are_predeclared_and_not_picked_by_gt(tmp_path):
    root=tmp_path
    source=root/"calibration"/"external"
    matrix=source/"matrix"
    matrix.mkdir(parents=True)
    for name in ("selected","extra"):
        np.savez(matrix/f"{name}.npz",
                 mtx_A=_K(),dist_A=np.zeros((1,5)))
    with (root/"calibration"/"external_calibration.csv").open(
            "w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=[
            "subject","main","main_link","path"])
        w.writeheader()
        w.writerow({"subject":"S1","main":"1","main_link":"False",
                    "path":"calibration/external/matrix/SS1_G12_selected.npz"})
        w.writerow({"subject":"S1","main":"6","main_link":"True",
                    "path":"calibration/external/matrix/SS1_G65_extra.npz"})
        # Source CSV can have a row for this group pointing at a different group.
        # Never select it merely because the main camera / pair matches.
        w.writerow({"subject":"S1","main":"6","main_link":"True",
                    "path":"calibration/external/matrix/SS23_G65_cross_group.npz"})
    # Create exact named files, no dynamic GT-dependent selection.
    (matrix/"selected.npz").rename(matrix/"SS1_G12_selected.npz")
    (matrix/"extra.npz").rename(matrix/"SS1_G65_extra.npz")
    K1,dist1=calibration_intrinsics(root,"S1","C1")
    K6,dist6=calibration_intrinsics(root,"S1","C6")
    assert np.allclose(K1,_K()) and np.allclose(K6,_K())
    assert dist1.size==5 and dist6.size==5
    with pytest.raises(ValueError):
        calibration_intrinsics(root,"S99","C1")


def test_model_only_PnP_test_recording_counts():
    import cv2
    xyz=_geometry()
    K=_K()
    scale=16
    pixels,_=cv2.projectPoints(
        xyz*scale,np.zeros(3),np.array([1.,-1.,40.]),K,np.zeros(5))
    target=xyz*scale+np.array([1.,-1.,40.])
    frames=[{
        "slot":i,
        "model_3d":xyz,
        "model_xy":pixels.reshape(-1,2),
        "reference_camera_xyz":target,
    } for i in range(24)]
    a=evaluate_recording(frames,K,np.zeros(5),scale,action="A3")
    b=evaluate_recording(frames,K,np.zeros(5),scale,action="A4")
    assert a["test_valid_six_joint_images"]==12
    assert b["test_valid_six_joint_images"]==24
    assert a["PnP_location_predictions_with_positive_depth"]==12
    assert b["PnP_location_predictions_with_positive_depth"]==24
    assert a["test_Vicon_used_in_prediction"] is False
    assert a["metric_MPJPE_mm_validated"] is False


def test_five_group_equal_weighting_and_missing_folds_rejected():
    rows=[]
    for camera in CAMERAS:
        for model in VARIANTS:
            for g in ANCHORS:
                def scores(n):
                    return {"n_people":n,
                        "valid_PnP_frames":9*n,
                        "valid_source_frames":10*n,
                        "group_mean_sixjoint_error_pct_femur":25.,
                        "group_mean_root_error_pct_femur":35.,
                        "median_model_self_reprojection_px":5.,
                        "camera_reference_projection_p90_px":
                            6. if camera=="C1" else 80.}
                rows.append({"camera":camera,"model":model,"heldout_group":g,
                    "A3":scores(4 if g in ("S1","S6") else 6),
                    "A4":scores(1)})
    out=aggregate_fold_rows(rows)
    assert len(out["cells"])==8
    assert out["new_independent_camera_calibrations"]==0
    assert out["no_test_Vicon_root_or_bone_length_used_for_prediction"]
    assert all(z["median_equal_group_sixjoint_error_pct_femur"]==25.
               for z in out["cells"])
    assert all(z["camera_source_intrinsics_evidence_warning"]
               for z in out["cells"] if z["camera"]=="C6")
    with pytest.raises(ValueError):
        aggregate_fold_rows(rows[:-1])
