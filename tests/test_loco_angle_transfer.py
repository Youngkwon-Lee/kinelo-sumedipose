"""Synthetic tests: prediction-only features, frozen-group weights, no GT test leakage."""
import numpy as np
import pytest

from sumedipose.loco_angle_transfer import (
    ANCHORS,CAMERAS,VARIANTS,BANDS,angle_band,sample_angles,
    group_equal_training,fit_bias,corrected_angle,error_metrics,evaluate_groups,
)


def _frames():
    return [(0,
      np.asarray([[1.,0,0],[0,0,0],[0,1.,0],[1,0,1.],[0,0,1.],[0,1.,1.]]),
      np.asarray([[1.,0,0],[0,0,0],[0,1.,0],[1,0,1.],[0,0,1.],[0,1.,1.]]))]


def test_observations_and_phase_bins_are_gt_for_reporting_only():
    obs=sample_angles(_frames())
    assert len(obs)==2 and obs[0]["p"]==pytest.approx(90)
    assert obs[0]["t"]==pytest.approx(90)
    assert angle_band(119.99)==BANDS[0]
    assert angle_band(120)==BANDS[1]
    assert angle_band(150)==BANDS[2]


def _train_rows(*, unequal=False):
    groups=[]
    for i,g in enumerate(ANCHORS):
        subjects=[f"synthetic-{i}-{j}" for j in range(4+i if unequal else 4)]
        for s in subjects:
            rows=[{"p":80.+n%6*20.,
                   "t":80.+n%6*20.-(i+1)*2.,
                   "bin":BANDS[0]} for n in range(20)]
            groups.append({"subject":s,"group":g,"A3":rows})
    return groups


def test_leave_one_group_out_weights_and_no_target_data():
    rows=_train_rows(unequal=True)
    selected=group_equal_training(rows,ANCHORS[0])
    assert len(selected)==sum((4+i)*20 for i in range(1,5))
    assert sum(w for _,_,w in selected)==pytest.approx(1.)
    expected_error=np.mean([4.,6.,8.,10.])
    assert fit_bias(selected,"constant")["intercept"]==pytest.approx(expected_error)
    # Moving heldout reference labels cannot change fit
    for item in rows:
        if item["group"]==ANCHORS[0]:
            for d in item["A3"]:d["t"]=-1000.
    selected2=group_equal_training(rows,ANCHORS[0])
    assert fit_bias(selected2,"constant")["intercept"]==pytest.approx(expected_error)


def test_slope_is_frozen_and_nonzero_and_shrunk():
    # Error increases as predicted angle decreases, using only p, t training
    inputs=[(p,10.-.8*((p-150.)/40.),1./240.) for p in
            [90.,130.,170.] for _ in range(80)]
    a=fit_bias(inputs,"constant")
    b=fit_bias(inputs,"linear_ridge")
    assert a["slope"]==0
    assert b["slope"]<0
    assert abs(b["slope"])<.8
    assert corrected_angle(90.,b)!=pytest.approx(corrected_angle(90.,a))


def test_no_gt_reference_allowed_in_prediction():
    inputs=[(float(90+i%5*20),float(i%9),1./140.)
            for i in range(140)]
    fit=fit_bias(inputs,"linear_ridge")
    x=corrected_angle(121.,fit)
    obs=[{"p":121.,"t":80.,"bin":BANDS[0]},
         {"p":121.,"t":155.,"bin":BANDS[2]}]
    assert corrected_angle(obs[0]["p"],fit)==pytest.approx(x)
    assert corrected_angle(obs[1]["p"],fit)==pytest.approx(x)
    result=error_metrics(obs,fit)
    assert result["n"]==2


def test_frozen_method_and_bad_training_fail_closed():
    with pytest.raises(ValueError,match="Choose"):
        fit_bias([], "raw")
    with pytest.raises(ValueError,match="Insufficient"):
        fit_bias([(80.,10.,1.)],"constant")
    with pytest.raises(ValueError,match="four"):
        group_equal_training(_train_rows()[:4], ANCHORS[0])


def _record(subject,group,action,camera,model):
    # Each group has 4,4,6,6,6 A3 test subjects; all five A4 anchors exist.
    obs=[{"p":float(80.+(j%9)*10.),
          "t":float(80.+(j%9)*10.-(group+1)*2),
          "bin":angle_band(float(80.+(j%9)*10.-(group+1)*2))}
          for j in range(24)]
    return {"subject":subject,"group":ANCHORS[group],
            "action":action,"camera":camera,"model":model,
            "observations":obs,"attempts":24,
            "detected":24,"valid_six_joint_frames":12}


def _synthetic_cohort():
    size=[4,4,6,6,6]
    mapping={}
    for i,g in enumerate(ANCHORS):
        mapping[g]=g
        for j in range(size[i]-1):mapping[f"synthetic-{i}-{j}"]=g
    records=[_record(s,ANCHORS.index(g),"A3",cam,model)
             for s,g in mapping.items() for cam in CAMERAS for model in VARIANTS]
    records.extend(_record(g,i,"A4",cam,model) for i,g in enumerate(ANCHORS)
                   for cam in CAMERAS for model in VARIANTS)
    return mapping,records


def test_26_and_five_anchors_stay_5_independent_folds():
    mapping,records=_synthetic_cohort()
    result,private=evaluate_groups(records,mapping)
    assert result["n_A3_subjects"]==26
    assert result["n_historical_calibration_groups"]==5
    assert result["model_recording_reports"]==124
    assert result["model_image_inferences"]==2976
    assert result["poses_detected"]==2976
    assert len(result["results"])==4*2*3
    assert len(private["detailed_calibration_group_metrics"])==5*4*2*3
    for group in private["detailed_calibration_group_metrics"]:
        assert group["train_calibration_groups"]==4
        assert group["train_subjects"] in (20,22)


def test_incomplete_source_group_is_rejected():
    mapping,records=_synthetic_cohort()
    with pytest.raises(ValueError,match="Incomplete"):
        evaluate_groups(records[:-1],mapping)


def test_very_large_bias_flags_out_of_range_without_clipping():
    obs=[{"p":170.,"t":80.,"bin":BANDS[0]}]
    large={"intercept":-100.,"slope":0.,"train_min_prediction":100.,
           "train_max_prediction":160.}
    report=error_metrics(obs,large)
    assert report["outside_0_to_180"]==1
    assert report["outside_train_pred_range"]==1
