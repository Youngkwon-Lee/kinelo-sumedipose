"""SUMediPose A3 26-subject leave-one-calibration-group-out angle correction.

Vicon-supervised and exploratory; NEVER medical or camera-only calibration.
All training subjects belong to the other FOUR historical calibration groups.
A3 subjects in the target group and its single A4 anchor are TEST ONLY.
Use exact prior source-valid RGB pairing and original frozen MediaPipe assets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from .geometry import joint_angle
from .position_holdout import ANCHORS, CAMERAS, VARIANTS, read_pose_frames

ACTION_A3, ACTION_A4 = "A3", "A4"
METHODS = ("raw", "constant", "linear_ridge")
EVAL_SETS = ("A3_unseen_group", "A4_unseen_group")
BANDS = ("under120", "120to150", "atleast150")
RIDGE_LAMBDA = 0.50  # fixed before looking at test results
PRED_CENTER_DEG = 150.
PRED_SCALE_DEG = 40.


def angle_band(reference: float) -> str:
    return BANDS[0] if reference < 120 else (
        BANDS[1] if reference < 150 else BANDS[2])


def sample_angles(frames: list[tuple]) -> list[dict]:
    """Build transient supervised observations, never write frame angles or XYZ."""
    records = []
    for slot, predicted, truth in frames:
        for indices in ((0, 1, 2), (3, 4, 5)):
            p = joint_angle(*(predicted[i] for i in indices))
            t = joint_angle(*(truth[i] for i in indices))
            if not math.isfinite(p) or not math.isfinite(t):
                raise ValueError("Non-finite knee included angle")
            records.append({"p": float(p), "t": float(t),
                            "bin": angle_band(t)})
    return records


def group_equal_training(rows: list[dict], heldout: str) -> list[tuple[float,float,float]]:
    """Weight: each of four training groups equally; each person within equally."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["group"] != heldout:
            by_group[row["group"]].append(row)
    if set(by_group) != set(ANCHORS) - {heldout}:
        raise ValueError("LOCO requires four fully distinct training calibration groups")
    weighted = []
    for group, people in by_group.items():
        if len({r["subject"] for r in people}) != len(people):
            raise ValueError("Training participant reused within one group")
        for row in people:
            if not row["A3"]:
                raise ValueError("Empty subject A3 training observations")
            weight = 1. / (4. * len(people) * len(row["A3"]))
            for obs in row["A3"]:
                weighted.append((float(obs["p"]),float(obs["p"]-obs["t"]),weight))
    if abs(sum(z[2] for z in weighted) - 1.) > 1e-10:
        raise AssertionError("Group-equal training weights do not sum to one")
    return weighted


def fit_bias(training: list[tuple[float,float,float]], method: str) -> dict:
    """Fit e = intercept + slope * ((predicted included angle-150)/40).

    Fixed ridge shrinkage on slope only; constant model has slope=0.
    Target Vicon reference labels are *never* supplied to predict().
    """
    if method not in METHODS[1:]:
        raise ValueError("Choose constant or linear_ridge")
    if len(training) < 100 or not all(
        math.isfinite(p) and math.isfinite(e) and math.isfinite(w) and w > 0
        for p,e,w in training):
        raise ValueError("Insufficient or invalid frozen A3 training observations")
    z = np.asarray([(p - PRED_CENTER_DEG)/PRED_SCALE_DEG for p,_,_ in training])
    y = np.asarray([e for _,e,_ in training])
    w = np.asarray([a for _,_,a in training])
    w = w / float(np.sum(w))
    zbar = float(np.sum(w*z))
    ybar = float(np.sum(w*y))
    variance = float(np.sum(w*(z-zbar)**2))
    covariance = float(np.sum(w*(z-zbar)*(y-ybar)))
    slope = (covariance/(variance+RIDGE_LAMBDA)
             if method == "linear_ridge" else 0.)
    intercept = ybar - slope*zbar
    if not all(math.isfinite(val) for val in (intercept,slope)):
        raise ValueError("Invalid fitted bias")
    return {"intercept":intercept,"slope":slope,
            "train_min_prediction":min(x[0] for x in training),
            "train_max_prediction":max(x[0] for x in training),
            "train_knee_angle_samples":len(training),
            "ridge_lambda":RIDGE_LAMBDA if method=="linear_ridge" else None,
            "method":method}


def corrected_angle(predicted: float, trained: dict | None) -> float:
    if trained is None:
        return predicted
    z = (predicted-PRED_CENTER_DEG)/PRED_SCALE_DEG
    return float(predicted-trained["intercept"]-trained["slope"]*z)


def error_metrics(obs: list[dict], fitted: dict | None) -> dict:
    if not obs:
        return {"n":0}
    base = [abs(v["p"]-v["t"]) for v in obs]
    adj = [corrected_angle(v["p"],fitted)-v["t"] for v in obs]
    return {"n":len(obs),
            "baseline_mae":mean(base),
            "corrected_mae":mean(abs(d) for d in adj),
            "improvement":mean(base)-mean(abs(d) for d in adj),
            "signed_error_after":mean(adj),
            "outside_0_to_180":sum(not 0 <= corrected_angle(v["p"],fitted)<=180 for v in obs),
            "outside_train_pred_range":(
                sum(v["p"]<fitted["train_min_prediction"] or
                    v["p"]>fitted["train_max_prediction"] for v in obs)
                if fitted is not None else None)}


def _prior_strict(analysis: Path, subject: str, camera: str,
                  action: str, model: str, asset_sha: str) -> dict:
    path = (analysis/"strict_source_pairing_20261117"/
            f"strict_{action}D2_{subject}_{camera}_{model}.json")
    if not path.is_file():
        raise ValueError("Missing previous strict source-valid recording report")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if (report["sample"]!=f"{camera}{subject}{action}D2" or
            report["model_variant"]!=model or
            report["model_asset_sha256"]!=asset_sha or
            report["pairing"]["new_reference_uses_dup_num_equals_factor"] is not True or
            report["pairing"]["changed_jpg_name_count"]!=0):
        raise ValueError("Prior strict identity/model/source-valid mapping mismatch")
    return report


def infer_record(root: Path, analysis: Path, subject: str, group: str, action: str,
                 camera: str, model: str, asset: Path, model_hash: str) -> dict:
    frames, qa = read_pose_frames(root, asset, action, camera, subject)
    previous = _prior_strict(analysis,subject,camera,action,model,model_hash)
    if (qa["model_images_attempted"]!=24 or
            qa["pose_detections"]!=previous["model_detected_poses"] or
            qa["previous_index_selection_invalid_pairs"]!=
            previous["pairing"]["original_wrong_coordinate_references"] or
            qa["selected_jpg_names_different_from_previous"]!=0 or
            qa["ambiguous_or_unverifiable_jpg_groups"]!=0):
        raise ValueError("Original 24-image source pairing/replay check failed")
    angles = sample_angles(frames)
    if len(angles)<12 or len(angles)!=2*qa["valid_six_joint_frames"]:
        raise ValueError("Too few verified six-visible-joint frames")
    return {"subject":subject,"group":group,"action":action,
            "camera":camera,"model":model,"observations":angles,
            "attempts":24,"detected":qa["pose_detections"],
            "valid_six_joint_frames":qa["valid_six_joint_frames"]}


def _group_summary(test_people: list[dict], fit: dict | None) -> dict:
    """Subjects equally weighted in each target calibration group."""
    if not test_people:
        return {"subjects_with_data":0,"n":0}
    per_person = [error_metrics(p["observations"],fit) for p in test_people]
    out = {"subjects_with_data":len(per_person),
           "n":sum(m["n"] for m in per_person),
           "baseline_mae":mean(m["baseline_mae"] for m in per_person),
           "corrected_mae":mean(m["corrected_mae"] for m in per_person),
           "improvement":mean(m["improvement"] for m in per_person),
           "out_of_range":sum(m["outside_0_to_180"] for m in per_person),
           "outside_train_pred_range":sum(
               m["outside_train_pred_range"] for m in per_person)
               if fit is not None else None}
    return out


def evaluate_groups(records: list[dict], mapping: dict[str,str]) -> tuple[dict,dict]:
    """Fit only A3 from four other groups, evaluate target A3 and A4."""
    expected_a3={(s,c,m) for s in mapping for c in CAMERAS for m in VARIANTS}
    expected_a4={(s,c,m) for s in ANCHORS for c in CAMERAS for m in VARIANTS}
    a3={(r["subject"],r["camera"],r["model"]):r for r in records if r["action"]=="A3"}
    a4={(r["subject"],r["camera"],r["model"]):r for r in records if r["action"]=="A4"}
    if (len(a3)!=104 or len(a4)!=20 or set(a3)!=expected_a3 or set(a4)!=expected_a4
        or len(records)!=124):
        raise ValueError("Incomplete A3 26-person/A4 five-anchor cohort")
    if {s:sum(z==s for z in mapping.values()) for s in ANCHORS}!=dict(
        zip(ANCHORS,(4,4,6,6,6))):
        raise ValueError("Frozen five-session source cohort changed")
    detailed=[]
    for camera in CAMERAS:
        for model in VARIANTS:
            all_a3=[{"subject":r["subject"],"group":r["group"],
                     "A3":r["observations"]}
                    for r in a3.values() if r["camera"]==camera and r["model"]==model]
            for heldout in ANCHORS:
                weighted=group_equal_training(all_a3,heldout)
                models={name:fit_bias(weighted,name) for name in METHODS[1:]}
                target_a3=[r for r in a3.values() if r["group"]==heldout and
                           r["camera"]==camera and r["model"]==model]
                target_a4=[a4[(heldout,camera,model)]]
                if len(target_a3)!=sum(g==heldout for g in mapping.values()):
                    raise ValueError("Heldout target group incomplete")
                for split,targets in ((EVAL_SETS[0],target_a3),
                                      (EVAL_SETS[1],target_a4)):
                    for method in METHODS:
                        fit=models.get(method)
                        allmetrics=_group_summary(targets,fit)
                        per_band={}
                        for band in BANDS:
                            filtered=[]
                            for row in targets:
                                q={**row,"observations":[o for o in row["observations"]
                                                       if o["bin"]==band]}
                                if q["observations"]:
                                    filtered.append(q)
                            per_band[band]=_group_summary(filtered,fit)
                        detailed.append({
                            "group":heldout,"camera":camera,"model":model,
                            "split":split,"method":method,
                            "target_subjects":len(targets),
                            "train_subjects":len(all_a3)-len(target_a3),
                            "train_calibration_groups":4,
                            "fit":fit and {k:round(v,6) if isinstance(v,float) else v
                                           for k,v in fit.items()},
                            "all":allmetrics,"by_band":per_band,
                        })
    aggregate=[]
    for camera in CAMERAS:
        for model in VARIANTS:
            for split in EVAL_SETS:
                raw_rows=[r for r in detailed if r["camera"]==camera and
                          r["model"]==model and r["split"]==split and
                          r["method"]==METHODS[0]]
                if len(raw_rows)!=5:raise ValueError("Missing baseline group")
                for method in METHODS:
                    selected=[r for r in detailed if r["camera"]==camera and
                        r["model"]==model and r["split"]==split and
                        r["method"]==method]
                    if len(selected)!=5 or {r["group"] for r in selected}!=set(ANCHORS):
                        raise ValueError("Missing independent held-out group in comparison")
                    def calculate(seq: list[dict]) -> dict:
                        if not seq:
                            return {"groups_with_data":0,"knee_angles":0}
                        vals=[x["all"] for x in seq]
                        return {
                            "groups_with_data":len(seq),
                            "target_subjects":sum(v["subjects_with_data"] for v in vals),
                            "knee_angles":sum(v["n"] for v in vals),
                            "mean_group_baseline_mae":round(mean(v["baseline_mae"] for v in vals),4),
                            "mean_group_corrected_mae":round(mean(v["corrected_mae"] for v in vals),4),
                            "mean_group_paired_improvement":round(mean(v["improvement"] for v in vals),4),
                            "median_group_paired_improvement":round(median(v["improvement"] for v in vals),4),
                            "groups_improved":sum(v["improvement"]>1e-7 for v in vals),
                            "groups_worsened":sum(v["improvement"]< -1e-7 for v in vals),
                            "corrected_angles_out_of_range":sum(v["out_of_range"] for v in vals),
                            "test_angles_outside_training_prediction_range":(
                                sum(v["outside_train_pred_range"] for v in vals) if method!="raw"
                                else None)}
                    row={"camera":camera,"model":model,"split":split,
                         "method":method,**calculate(selected)}
                    row["by_vicon_reference_angle"]={}
                    for band in BANDS:
                        part=[{"all":q["by_band"][band]} for q in selected
                              if q["by_band"][band]["n"]>0]
                        row["by_vicon_reference_angle"][band]=calculate(part)
                    aggregate.append(row)
    overview={
        "status":"complete_exploratory_A3_26_A4_5_LOCO_angle_transfer",
        "source_main_code_start_sha":"52b7d92635f3a942c9c06e96f471977e50ffe29f",
        "n_A3_subjects":26,"n_A4_anchor_subjects":5,
        "n_historical_calibration_groups":5,"new_independent_calibration_groups":0,
        "A3_subjects_in_test_group_never_in_training":True,
        "A4_test_labels_never_used_for_training":True,
        "train":"A3 D2 24 strict-source-valid images per subject, from other four groups",
        "test":["heldout group's A3 D2 24 images (new participants vs train)",
                "heldout group's A4 D2 24 images (new action vs train)"],
        "models":list(VARIANTS),"views":list(CAMERAS),
        "methods":{
            "raw":"no bias correction",
            "constant":"group-equal training signed-angle mean",
            "linear_ridge":"fixed one-feature residual slope, centered predicted included angle 150°, scale 40°, ridge slope λ=0.50 (not tuned)"},
        "model_recording_reports":len(records),
        "model_image_inferences":sum(r["attempts"] for r in records),
        "poses_detected":sum(r["detected"] for r in records),
        "valid_six_joint_frames":sum(r["valid_six_joint_frames"] for r in records),
        "results":aggregate,
        "caveats":{
            "true_camera_Vicon_exposure_sync_verified":False,
            "independent_anatomically_equivalent_lm_ground_truth":False,
            "heldout_groups_independent_of_historical_model_development":False,
            "new_capture_calibration_days":0,
            "regularization_is_predeclared_not_optimized_on_test":True,
            "Vicon_reference_supervised_method_not_clinically_deployable":True,
            "target_ref_angle_used_only_for_reporting_bands_not_prediction":True,
            "physical_MPJPE_mm_validated":False,
            "no_original_RGB_XYZ_or_per_frame_prediction_written":True,
        },
    }
    private={
        "status":overview["status"],
        "detailed_calibration_group_metrics":detailed,
        "private_subject_audit":[{
            "subject":r["subject"],"group":r["group"],
            "camera":r["camera"],"model":r["model"],"action":r["action"],
            "n_valid_angle":len(r["observations"]),
            "n_poses_detected":r["detected"],
            "n_six_joint_frames":r["valid_six_joint_frames"],
        } for r in records],
        "image_frame_coordinates_or_raw_angles_exported":False,
    }
    return overview,private


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--analysis-dir",type=Path,required=True)
    ap.add_argument("--private-output",type=Path,required=True)
    ap.add_argument("--aggregate-output",type=Path,required=True)
    args=ap.parse_args(argv)
    repo=Path(__file__).resolve().parents[2]
    analysis=args.analysis_dir.resolve()
    outputs=[args.private_output.resolve(),args.aggregate_output.resolve()]
    if len(set(outputs))!=2 or any(p.exists() or p.parent!=analysis or
       p==repo or repo in p.parents for p in outputs):
        ap.error("Write two new files only under approved private research analysis dir")
    census=json.loads((analysis/"A3_26subject_D2_census_protocol_20261116.json"
                      ).read_text(encoding="utf-8-sig"))
    mapping=json.loads((args.root/"calibration"/"external"/"subject_map.json"
                      ).read_text(encoding="utf-8-sig"))
    subject_group={s:mapping[s] for s in mapping if s not in census["missing_subjects"]}
    if (set(subject_group)!={f"S{i}" for i in range(1,29)}-set(census["missing_subjects"])
        or len(subject_group)!=26 or
        {g:sum(v==g for v in subject_group.values()) for g in ANCHORS}
        !=census["calibration_group_sizes"] or
        any(subject_group[s]!=s for s in ANCHORS)):
        ap.error("Actual source subject map does not match frozen original 26-person census")
    assets={}
    for v in VARIANTS:
        p=args.model_dir/f"pose_landmarker_{v}.task"
        digest=sha256(p.read_bytes()).hexdigest()
        if digest!=census["model_variants"][v]["sha256"]:
            ap.error("Frozen MediaPipe model hash mismatch")
        assets[v]=(p,digest)
    records=[]
    for camera in CAMERAS:
        for model in VARIANTS:
            path,digest=assets[model]
            for subject in subject_group:
                records.append(infer_record(
                    args.root,analysis,subject,subject_group[subject],
                    ACTION_A3,camera,model,path,digest))
            for subject in ANCHORS:
                records.append(infer_record(
                    args.root,analysis,subject,subject_group[subject],
                    ACTION_A4,camera,model,path,digest))
    overview,private=evaluate_groups(records,subject_group)
    private["script_sha256"]=sha256(Path(__file__).read_bytes()).hexdigest()
    for p,content in zip(outputs,(private,overview)):
        with p.open("x",encoding="utf8") as f:
            json.dump(content,f,ensure_ascii=False,indent=2)
            f.write("\n")
    print(json.dumps({
        "status":overview["status"],"inferences":overview["model_image_inferences"],
        "detected":overview["poses_detected"],"reports":overview["model_recording_reports"],
        "results":[{k:v[k] for k in ("camera","model","split","method",
            "mean_group_baseline_mae","mean_group_corrected_mae",
            "mean_group_paired_improvement","groups_improved","groups_worsened")}
            for v in overview["results"] if v["split"]=="A4_unseen_group"],
    }))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
