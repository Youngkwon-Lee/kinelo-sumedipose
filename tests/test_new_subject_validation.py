"""Only synthetic per-person reports; never access original SUMediPose images."""
import copy

import pytest

from sumedipose.new_subject_validation import grouped_validation


def fake_fixtures():
    groups=[{
        "calibration_group":f"K{i}",
        "subjects":[f"X{i}a",f"X{i}b"],
    } for i in range(1,6)]
    protocol={
        "subject_groups":groups, "preexisting_pilot_subjects":[f"Z{i}" for i in range(1,6)],
        "repeats":["D1","D2","D3"],"cameras":["C1","C6"],
        "unique_image_samples_per_recording":24,
        "expected_aggregate_reports":120,
        "expected_image_inferences":2880,
        "models":{
            "lite":{"sha256":"lite-test-hash"},
            "full":{"sha256":"full-test-hash"},
        },
    }
    cal={"comparisons":[{"session":f"K{i}","pair":"G65","sampson_median_px":float(i)}
                        for i in range(1,6)]}
    old={"status":"complete_exploratory","complete_session_count":5,
         "associations":[
             {"camera":cam,"model":model,"reference_space":mod,"spearman_rho_descriptive":.5}
             for cam in ("C1","C6") for model in ("lite","full")
             for mod in ("image_2d","world_3d")
         ]}
    reports={}
    for i,g in enumerate(groups,1):
        for sub in g["subjects"]:
            for rep in ("D1","D2","D3"):
                for cam in ("C1","C6"):
                    for model in ("lite","full"):
                        val=float(i+(cam=="C6")*2+(model=="full")*.5)
                        reports[(sub,rep,cam,model)]={
                            "sample":f"{cam}{sub}A3{rep}",
                            "model_variant":model,
                            "model_asset_sha256":protocol["models"][model]["sha256"],
                            "unique_images_sampled":24,
                            "max_unique_images_requested":24,
                            "inference_mode":"IMAGE",
                            "model_detected_poses":24,
                            "results":{mod:{
                                "left":{"valid":24,"MAE_degrees":val},
                                "right":{"valid":24,"MAE_degrees":val},
                            } for mod in ("image_2d","world_3d")},
                        }
    return protocol,cal,old,reports


def test_repeat_and_subject_grouping_never_inflate_calibration_n():
    p,c,o,reports=fake_fixtures()
    r=grouped_validation(p,c,o,reports)
    assert r["status"] == "complete_subject_generalization_exploratory"
    assert r["new_subject_codes"] == 10
    assert r["total_distinct_calibration_groups"] == 5
    assert r["new_independent_calibration_sessions"] == 0
    assert r["recordings"] == 30
    assert r["model_evaluation_reports"] == 120
    assert r["image_inferences"] == 2880
    assert r["pose_detections"] == 2880
    assert len(r["numeric_results_by_camera_model_reference_space"]) == 8
    assert all(x["n_calibration_groups"]==5 for x in r["numeric_results_by_camera_model_reference_space"])
    assert all(x["new_subject_group_level_spearman_rho"] == 1 for x in r["numeric_results_by_camera_model_reference_space"])


def test_missing_report_fails_closed():
    p,c,o,reports=fake_fixtures()
    reports.pop(next(iter(reports)))
    with pytest.raises(ValueError,match="missing=1"):
        grouped_validation(p,c,o,reports)


def test_model_sha_and_sampling_identity_rejected():
    p,c,o,reports=fake_fixtures()
    first=next(iter(reports))
    bad=copy.deepcopy(reports)
    bad[first]["model_asset_sha256"]="mismatch"
    with pytest.raises(ValueError,match="SHA256"):
        grouped_validation(p,c,o,bad)
    bad=copy.deepcopy(reports)
    bad[first]["unique_images_sampled"]=23
    with pytest.raises(ValueError,match="sampling"):
        grouped_validation(p,c,o,bad)
    bad=copy.deepcopy(reports)
    bad[first]["sample"]="C1S1A3D1"
    with pytest.raises(ValueError,match="identity"):
        grouped_validation(p,c,o,bad)


def test_reject_tuned_participant_and_wrong_calibration_root():
    p,c,o,reports=fake_fixtures()
    p=copy.deepcopy(p)
    p["preexisting_pilot_subjects"].append("X1a")
    with pytest.raises(ValueError,match="Previously"):
        grouped_validation(p,c,o,reports)
    p,c,o,reports=fake_fixtures()
    c["comparisons"][0]["session"]="unmatched"
    with pytest.raises(ValueError,match="mapping"):
        grouped_validation(p,c,o,reports)


def test_invalid_knee_angles_fail_no_fabricated_completeness():
    p,c,o,reports=fake_fixtures()
    key=next(iter(reports))
    for side in ("left","right"):
        reports[key]["results"]["world_3d"][side]={"valid":0,"MAE_degrees":None}
    with pytest.raises(ValueError,match="without valid"):
        grouped_validation(p,c,o,reports)
