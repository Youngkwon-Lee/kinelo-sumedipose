"""Local-only grouped new-participant validation of SUMediPose A3 pose MAE.

Aggregate three repeated squats within each new participant and two new
participants within each of the five existing calibration groups. Never treat
images, repeats or participants sharing one calibration as independent
calibration sessions. No images, per-frame or participant-level data are
included in reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from .calibration_strata import spearman_descriptive, weighted_two_side_mae

CAMERAS = ("C1", "C6")
MODELS = ("lite", "full")
MODALITIES = ("image_2d", "world_3d")
REPEATS = ("D1", "D2", "D3")


def _validate_protocol(protocol, calibration, previous):
    groups = protocol["subject_groups"]
    if (len(groups) != 5 or len({g["calibration_group"] for g in groups}) != 5
            or any(len(g["subjects"]) != 2 for g in groups)
            or len({s for g in groups for s in g["subjects"]}) != 10):
        raise ValueError("Expected five unique calibration groups, each with two new participants")
    if (tuple(protocol["repeats"]) != REPEATS
            or tuple(protocol["cameras"]) != CAMERAS
            or protocol["unique_image_samples_per_recording"] != 24
            or protocol["expected_aggregate_reports"] != 120
            or protocol["expected_image_inferences"] != 2880):
        raise ValueError("Incompatible fixed 24-image, 3-repeat validation protocol")
    anchor = set(protocol["preexisting_pilot_subjects"])
    if any(s in anchor for g in groups for s in g["subjects"]):
        raise ValueError("Previously evaluated participants must not be in new-person set")
    if previous.get("status") != "complete_exploratory" or previous.get("complete_session_count") != 5:
        raise ValueError("Original five-anchor analysis is not complete")
    calib = {}
    for row in calibration["comparisons"]:
        if row["pair"] != "G65":
            continue
        session = row["session"]
        if session in calib:
            raise ValueError("Duplicate G65 calibration session")
        val = float(row["sampson_median_px"])
        if not 0 <= val < float("inf"):
            raise ValueError("Invalid stereo residual")
        calib[session] = val
    if set(calib) != {g["calibration_group"] for g in groups}:
        raise ValueError("Calibration group mapping differs from original five sessions")
    if set(protocol["models"]) != set(MODELS):
        raise ValueError("Unexpected model set")
    return groups, calib


def grouped_validation(protocol, calibration, previous, reports):
    """Aggregate new subjects without exposing their identities in the result."""
    groups, cal = _validate_protocol(protocol, calibration, previous)
    expected_keys = {
        (subject, rep, cam, model)
        for group in groups for subject in group["subjects"]
        for rep in REPEATS for cam in CAMERAS for model in MODELS
    }
    if set(reports) != expected_keys:
        missing = len(expected_keys - set(reports))
        unexpected = len(set(reports) - expected_keys)
        raise ValueError(f"120 model reports required; missing={missing}, unexpected={unexpected}")

    measured = {}
    detections = 0
    images = 0
    valid_counts = {(mod, cam, model):0
                    for mod in MODALITIES for cam in CAMERAS for model in MODELS}
    for (subject, rep, cam, model), record in reports.items():
        if (record["sample"] != f"{cam}{subject}A3{rep}"
                or record["model_variant"] != model):
            raise ValueError("Unexpected model report identity")
        if record["model_asset_sha256"] != protocol["models"][model]["sha256"]:
            raise ValueError("Model SHA256 differs from frozen protocol")
        if (record["unique_images_sampled"] != 24
                or record["max_unique_images_requested"] != 24
                or record["inference_mode"] != "IMAGE"):
            raise ValueError("Frozen inference sampling protocol mismatch")
        d = int(record["model_detected_poses"])
        if not 0 <= d <= 24:
            raise ValueError("Invalid pose detection count")
        detections += d
        images += 24
        for mod in MODALITIES:
            mae, n = weighted_two_side_mae(record, mod, expected_frames=24)
            if n <= 0 or mae is None:
                raise ValueError("Session/view/model without valid joint-angle measurements")
            valid_counts[(mod,cam,model)] += n
            measured[(subject,rep,cam,model,mod)] = mae
    if images != 2880:
        raise ValueError("Wrong total inference sample count")

    group_means = {}
    within_subject_ranges = {}
    for group in groups:
        session = group["calibration_group"]
        for cam in CAMERAS:
            for model in MODELS:
                for mod in MODALITIES:
                    per_subject = []
                    for subject in group["subjects"]:
                        repeat_maes = [measured[(subject,rep,cam,model,mod)] for rep in REPEATS]
                        per_subject.append(mean(repeat_maes))
                        within_subject_ranges[(session,subject,cam,model,mod)] = (
                            max(repeat_maes)-min(repeat_maes)
                        )
                    group_means[(session,cam,model,mod)] = mean(per_subject)
    sessions = [g["calibration_group"] for g in groups]
    x = [cal[s] for s in sessions]
    comparisons = []
    for cam in CAMERAS:
        for model in MODELS:
            for mod in MODALITIES:
                y = [group_means[(s,cam,model,mod)] for s in sessions]
                rho = spearman_descriptive(x,y)
                loo = [spearman_descriptive(
                    [v for j,v in enumerate(x) if j != i],
                    [v for j,v in enumerate(y) if j != i])
                    for i in range(5)]
                prior = [z["spearman_rho_descriptive"] for z in previous["associations"]
                         if (z["camera"],z["model"],z["reference_space"]) == (cam,model,mod)]
                if len(prior) != 1:
                    raise ValueError("Expected one original pilot correlation per outcome")
                by_repeat = {
                    rep:spearman_descriptive(x,[
                        mean(measured[(sub,rep,cam,model,mod)] for sub in g["subjects"])
                        for g in groups
                    ]) for rep in REPEATS
                }
                ranges = [
                    within_subject_ranges[(s,sub,cam,model,mod)]
                    for g in groups for s in (g["calibration_group"],) for sub in g["subjects"]
                ]
                comparisons.append({
                    "camera":cam,"model":model,"reference_space":mod,
                    "n_calibration_groups":5,"n_new_subjects":10,
                    "n_repeat_measurements_per_subject":3,
                    "new_subject_group_level_spearman_rho":rho,
                    "original_anchor_subject_spearman_rho":prior[0],
                    "loo_group_rho_min":min(loo),"loo_group_rho_max":max(loo),
                    "D1_D2_D3_repeat_specific_rho":by_repeat,
                    "median_mae_across_five_groups_deg":round(median(y),4),
                    "median_within_subject_repeat_mae_range_deg":round(median(ranges),4),
                    "valid_joint_angle_measurements":valid_counts[(mod,cam,model)],
                })
    contrasts = []
    for model in MODELS:
        for mod in MODALITIES:
            diff=[group_means[(s,"C6",model,mod)]-group_means[(s,"C1",model,mod)]
                  for s in sessions]
            contrasts.append({
                "model":model,"reference_space":mod,
                "median_c6_minus_c1_mae_degrees":round(median(diff),4),
                "g65_vs_c6_minus_c1_rho":spearman_descriptive(x,diff),
                "view_difference_is_not_a_causal_adjustment":True,
            })
    return {
        "status":"complete_subject_generalization_exploratory",
        "action":"A3 squat only",
        "new_subject_codes":10,"total_distinct_calibration_groups":5,
        "new_independent_calibration_sessions":0,"recordings":30,
        "model_evaluation_reports":120,"image_inferences":images,
        "pose_detections":detections,
        "detection_fraction":round(detections/images,5),
        "all_model_sha_and_repeats_verified":True,
        "aggregation_unit":"valid-side weighted knee-angle MAE per recording -> equal mean across D1/D2/D3 within participant -> equal mean across two new participants within original calibration group -> five independent calibration groups",
        "comparison_data_reference":"S1A3_calibration_stratified_pose_overview.json",
        "calibration_reference":"S1_S6_S11_S17_S23_calibration_five_session_overview.json",
        "heldout_subjects_selected_before_model_execution":True,
        "numeric_results_by_camera_model_reference_space":comparisons,
        "paired_camera_view_contrasts":contrasts,
        "caveats":[
            "10 new subjects, but zero new calibration sessions: correlation n=5 groups, not n=15",
            "Descriptive only: no p-values, inferential significance or causality",
            "Source group G65 stereo epipolar residual is not single-view MediaPipe accuracy",
            "2D projected-Vicon and 3D WCS references are different geometric definitions",
            "MediaPipe landmarks differ anatomically from Vicon joint centers",
            "D1/D2/D3 are repeats of squat A3, not different exercise actions",
            "No independent exposure time/Vicon absolute 3D accuracy/model-training overlap verification",
        ],
        "raw_images_or_subject_level_results_exported":False,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    folder=args.analysis_dir
    protocol=json.loads((folder/"A3_new_subject_generalization_protocol_20261116.json").read_text(encoding="utf8"))
    calibration=json.loads((folder/"S1_S6_S11_S17_S23_calibration_five_session_overview.json").read_text(encoding="utf8"))
    previous=json.loads((folder/"S1A3_calibration_stratified_pose_overview.json").read_text(encoding="utf8"))
    reports={}
    for g in protocol["subject_groups"]:
        for sub in g["subjects"]:
            for rep in REPEATS:
                for cam in CAMERAS:
                    for model in MODELS:
                        key=(sub,rep,cam,model)
                        path=folder/f"A3_generalization_{sub}_{rep}_{cam}_{model}.json"
                        if not path.is_file():
                            parser.error(f"Missing private model result: {path.name}")
                        reports[key]=json.loads(path.read_text(encoding="utf8"))
    result=grouped_validation(protocol,calibration,previous,reports)
    repo=Path(__file__).resolve().parents[2]
    target=args.output.resolve()
    if target==repo or repo in target.parents:
        parser.error("Refuse to write participant-derived report inside the public repository")
    if target.exists():
        parser.error("Refuse to overwrite existing analysis file")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({key:result[key] for key in ("status","new_subject_codes","total_distinct_calibration_groups","model_evaluation_reports","image_inferences")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
