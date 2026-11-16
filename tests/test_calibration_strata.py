"""Synthetic subject/session fixtures only; no SUMediPose original images."""
import copy

from sumedipose.calibration_strata import analyse, spearman_descriptive

SES = ("S1","S6","S11","S17","S23")
RES = (6.77, 11.42, 1.29, 1.41, .51)


def fixtures():
    calibration = {"comparisons": [
        {"session":s,"pair":"G65","sampson_median_px":v}
        for s,v in zip(SES,RES)
    ]}
    protocol = {
        "repeat":"D2","max_unique_frames_per_view":24,
        "model_variants":{v:{"sha256":v*20} for v in ("lite","full")},
    }
    reports = {}
    for idx,s in enumerate(SES):
        for cam in ("C1","C6"):
            for variant in ("lite","full"):
                mae = 5+idx if cam == "C6" else 2+idx/10
                reports[(s,cam,variant)] = {
                    "sample":f"{cam}{s}A3D2","model_variant":variant,
                    "model_asset_sha256":variant*20,
                    "inference_mode":"IMAGE","unique_images_sampled":24,
                    "max_unique_images_requested":24,
                    "results":{
                        modality:{
                            "left":{"MAE_degrees":mae,"valid":20},
                            "right":{"MAE_degrees":mae+2,"valid":10},
                        } for modality in ("image_2d","world_3d")
                    },
                }
    return calibration,protocol,reports


def test_spearman_ties_and_constant():
    assert spearman_descriptive([1,2,3],[2,4,6]) == 1
    assert spearman_descriptive([1,2,3],[6,4,2]) == -1
    assert spearman_descriptive([1,2,3],[9,9,9]) is None


def test_all_five_and_no_frame_pseudoreplication():
    calibration,protocol,reports = fixtures()
    result = analyse(calibration,reports,protocol)
    assert result["complete_session_count"] == 5
    assert result["status"] == "complete_exploratory"
    assert len(result["associations"]) == 8
    assert len(result["camera_view_contrasts"]) == 4
    assert all(row["session_count"]==5 for row in result["associations"]+result["camera_view_contrasts"])
    assert all(row["group_comparison_validated"] is False for row in result["associations"])
    assert result["associations"][0]["posthoc_gt_cutoff_count"] == 2
    assert result["sessions_used_for_roi_tuning"] == ["S1"]


def test_missing_s1_is_explicit_and_five_session_claim_blocked():
    calibration,protocol,reports = fixtures()
    reports = {k:v for k,v in reports.items() if k[0] != "S1"}
    result = analyse(calibration,reports,protocol)
    assert result["status"] == "incomplete_missing_or_invalid_sessions"
    assert result["complete_session_count"] == 4
    assert len(result["missing_pose_report_files"]) == 4
    assert all(row["session_count"] == 4 for row in result["associations"])
    assert result["associations"][0]["posthoc_gt_cutoff_mae_median_deg"] is None


def test_asset_hash_or_sample_mismatch_rejected():
    import pytest
    calibration,protocol,reports = fixtures()
    bad = copy.deepcopy(reports)
    bad[("S1","C6","lite")]["model_asset_sha256"] = "other"
    with pytest.raises(ValueError,match="SHA"):
        analyse(calibration,bad,protocol)
    bad = copy.deepcopy(reports)
    bad[("S1","C6","lite")]["sample"] = "C1S1A3D2"
    with pytest.raises(ValueError,match="Mismatched"):
        analyse(calibration,bad,protocol)


def test_zero_valid_does_not_invent_mae_or_five_session_completion():
    calibration,protocol,reports = fixtures()
    for side in ("left","right"):
        reports[("S1","C6","lite")]["results"]["world_3d"][side] = {
            "MAE_degrees":None,"valid":0
        }
    result = analyse(calibration,reports,protocol)
    row = next(x for x in result["associations"]
               if x["model"]=="lite" and x["camera"]=="C6"
               and x["reference_space"]=="world_3d")
    assert row["session_count"] == 4
    assert result["status"] == "incomplete_missing_or_invalid_sessions"
