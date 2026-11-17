"""Synthetic-only leave-one-calibration-group-out 3D transfer regression tests."""
import numpy as np
import pytest

from sumedipose.position_holdout import ANCHORS, CAMERAS, VARIANTS, root_relative
from sumedipose.cross_person_3d_transfer import (
    TOTAL_SUBJECTS, PARITIES, validate_groups, training_only_fits,
    evaluate_excluded_group, summarize, _only_a3_last12,
)


def groups():
    # Synthetic source-like code/group map; no actual study data.
    g = {
        "S1": ("S1","S2","S3","S4"),
        "S6": ("S6","S7","S8","S9"),
        "S11": ("S11","S12","S13","S14","S15","S16"),
        "S17": ("S17","S18","S19","S20","S21","S22"),
        "S23": ("S23","S24","S25","S26","S27","S28"),
    }
    mapping = {s: root for root, members in g.items() for s in members}
    return g, mapping


def skeleton(index):
    b=np.array([
        [-.1,.05,.02],[-.13,-.37,.08],[-.12,-.72,.06],
        [.1,-.05,-.02],[.14,-.41,-.08],[.12,-.77,-.06],
    ])
    b[1,0]-=index*.003
    b[4,2]+=index*.002
    return root_relative(b)


def fake_frames():
    frame_map={}
    for sub in TOTAL_SUBJECTS:
        x=[skeleton(i+int(sub[1:])%7) for i in range(24)]
        y=[a @ np.diag([-1.,1.,1.])*800 for a in x]
        frame_map[sub]={
            "A3": ([(i,a,b) for i,(a,b) in enumerate(zip(x,y))],{})
        }
        if sub in ANCHORS:
            frame_map[sub]["A4"]=(
                [(i,a,b) for i,(a,b) in enumerate(zip(x,y))],{})
    return frame_map


def test_original_26_subjects_unique_and_group_balanced():
    grp,map_=groups()
    census={"missing_subjects":["S5","S10"],
            "calibration_group_sizes":{g:len(s) for g,s in grp.items()}}
    out=validate_groups(map_,census)
    assert out==grp
    assert len(set(s for subjects in out.values() for s in subjects))==26


def test_new_group_or_missing_source_cohort_rejected():
    grp,map_=groups()
    census={"missing_subjects":["S5","S10"],
            "calibration_group_sizes":{g:len(s) for g,s in grp.items()}}
    map_["S4"]="OTHER"
    with pytest.raises(ValueError,match="Original 26"):
        validate_groups(map_,census)


def test_train_only_parity_selection_and_cross_person_holdout():
    grp,_=groups()
    results=evaluate_excluded_group(fake_frames(),grp["S1"],"S1")
    assert set(results["train_anchor_groups"])==set(ANCHORS)-{"S1"}
    assert results["parity_chosen_from_train_only"]=="reflected"
    assert results["test_group_was_never_used_for_fit"]
    assert len(results["private_test_participant_rows"])==5
    a3=[r for r in results["private_test_participant_rows"]
        if r["evaluation"]=="A3_heldout"]
    a4=[r for r in results["private_test_participant_rows"]
        if r["evaluation"]=="A4_cross_action"]
    assert len(a3)==4 and len(a4)==1
    assert all(r["valid_test_frames"]==12 for r in a3)
    assert a4[0]["valid_test_frames"]==24
    assert all(r["scores"]["reflected"]==0 for r in a3+a4)


def test_training_uses_others_only():
    frames=fake_frames()
    wrong=tuple(["S1","S6"])
    with pytest.raises(ValueError,match="Training person"):
        evaluate_excluded_group(frames,wrong,"S1")


def test_insufficient_training_frames_fails_closed():
    frames=fake_frames()
    frames["S6"]["A3"]=(frames["S6"]["A3"][0][:4],{})
    grp,_=groups()
    with pytest.raises(ValueError,match="Training calibration anchor"):
        evaluate_excluded_group(frames,grp["S1"],"S1")


def synthetic_folds():
    grp,_=groups()
    rows=[]
    for cam in CAMERAS:
        for model in VARIANTS:
            for group in ANCHORS:
                prefer="reflected" if cam=="C1" else "proper"
                fit={"proper":31.,"reflected":20.} if prefer=="reflected" else {
                    "proper":22., "reflected":28.}
                scores={"proper":34.,"reflected":21.} if cam=="C1" else {
                    "proper":25.,"reflected":30.}
                test=[
                    {"subject":s,"evaluation":"A3_heldout","valid_test_frames":12,
                     "scores":dict(scores),"selected_training_only":scores[prefer]}
                    for s in grp[group]
                ]
                test.append({
                    "subject":group,"evaluation":"A4_cross_action",
                    "valid_test_frames":24,
                    "scores":dict(scores),"selected_training_only":scores[prefer]
                })
                rows.append({
                    "camera":cam,"model":model,"heldout_group":group,
                    "train_anchor_groups":tuple(g for g in ANCHORS if g!=group),
                    "train_valid_frames":48,
                    "train_fit_pct_femur":fit,
                    "parity_chosen_from_train_only":prefer,
                    "private_test_participant_rows":test,
                    "test_group_was_never_used_for_fit":True,
                })
    return rows,grp


def test_aggregate_uses_5_groups_not_26_independent_calibrations():
    rows,group_to_subjects=synthetic_folds()
    z=summarize(rows,group_to_subjects)
    assert len(z["cells"])==8
    assert z["calibration_groups"]==5
    assert z["new_independent_calibration_groups"]==0
    assert z["A3_subjects_evaluated"]==26
    assert z["heldout_model_participant_action_evaluations"]==124
    assert z["no_holdout_group_subject_or_session_in_fit"] is True
    assert z["GT_hip_root_uses_Vicon_test_landmarks"] is True
    assert z["not_absolute_MPJPE_mm_or_clinical_accuracy"] is True
    for row in z["cells"]:
        assert row["heldout_participants"]==(26 if row["evaluation"]=="A3_heldout" else 5)
        assert row["independent_heldout_calibration_groups"]==5
        if row["camera"]=="C1":
            assert row["median_heldout_group_mean_training_selected_pct_femur"]==21.
            assert row["training_selected_parity_count"]["reflected"]==5
        else:
            assert row["median_heldout_group_mean_training_selected_pct_femur"]==25.
            assert row["training_selected_parity_count"]["proper"]==5


def test_missing_fold_and_parity_test_leakage_rejected():
    rows,grp=synthetic_folds()
    with pytest.raises(ValueError,match="Missing or duplicate"):
        summarize(rows[:-1],grp)
    rows,grp=synthetic_folds()
    rows[0]["parity_chosen_from_train_only"]="proper"
    with pytest.raises(ValueError,match="train data"):
        summarize(rows,grp)


def test_borrowed_subject_in_training_leakage_rejected():
    rows,grp=synthetic_folds()
    rows[0]["train_anchor_groups"]=tuple(ANCHORS[:4])
    with pytest.raises(ValueError,match="contaminated"):
        summarize(rows,grp)


def test_other_person_fitted_baseline_not_required_for_full26():
    rows,grp=synthetic_folds()
    z=summarize(rows,grp)
    assert all(not any(k.startswith("personal_A3_fitted_anchor_median") for k in r)
               for r in z["cells"])
