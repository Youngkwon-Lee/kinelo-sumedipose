"""Source-role calibration provenance and S17/C6 PnP sensitivity (local-only).

This is a predeclared comparison of official S17/C6 CSV rows:
  * explicit cam=6/main=6 G65 row (path filename refers to S23)
  * alternate cam=5/main=6 G65 row (path filename refers to S17)
The first is the source-role candidate. We NEVER choose a matrix based on
test Vicon error. The previous private 26-person PnP result used the alternate
due to a filename group restriction, so only S17/C6 is re-inferred here.

Do not export raw camera matrices, subject-level scores, model weight files,
original images, Vicon trajectories or inferred 3D joint positions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from .camera_pnp_no_test_vicon import (
    calibration_intrinsics_with_provenance, camera_projection_consistency,
    read_inference, test_recording
)
from .position_holdout import VARIANTS, ANCHORS

TARGET_GROUP = "S17"
CAMERA = "C6"
MAIN_ROLE = "6"
PAIRED_ROLE = "5"
PAIR = "G65"
TARGET_A3_SUBJECTS = tuple(f"S{i}" for i in range(17, 23))
MIN_VALID = 6
EXPECTED_ORIGINAL_MODEL_INFERENCES = 2976


def source_role_candidates(root: Path):
    """Read only explicit S17/C6 source CSV rows; fail on ambiguous roles."""
    file = root/"calibration"/"external_calibration.csv"
    with file.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    valid = [
        r for r in rows if r.get("subject") == TARGET_GROUP
        and str(r.get("main")) == MAIN_ROLE
        and f"_{PAIR}_" in str(r.get("path", ""))
    ]
    out = {}
    for role in (MAIN_ROLE, PAIRED_ROLE):
        role_rows = [r for r in valid if str(r.get("cam")) == role]
        paths = {str(r["path"]) for r in role_rows}
        if len(paths) != 1:
            raise ValueError("Missing or multiple source CSV paths for camera role")
        rel = next(iter(paths))
        path = (root/rel).resolve()
        if (root.resolve() not in path.parents or
                path.suffix != ".npz" or not path.is_file()):
            raise ValueError("Source camera calibration path invalid")
        with np.load(path, allow_pickle=False) as m:
            K, dist = np.asarray(m["mtx_A"],float), np.asarray(m["dist_A"],float)
        if (K.shape != (3,3) or dist.size not in (4,5,8,12,14)
                or not np.isfinite(K).all() or not np.isfinite(dist).all()):
            raise ValueError("Malformed intrinsics in original CSV")
        out[role] = (K,dist,{
            "csv_camera_role":role,
            "filename_matches_declared_S17_group":
                Path(rel).name.startswith("SS17_G65_"),
            "source_provenance_selection_is_predeclared":True,
        })
    if (out[MAIN_ROLE][2]["filename_matches_declared_S17_group"]
            or not out[PAIRED_ROLE][2]["filename_matches_declared_S17_group"]):
        raise ValueError("Original S17 main-vs-paired camera provenance changed")
    K,d,meta = calibration_intrinsics_with_provenance(root,TARGET_GROUP,CAMERA)
    if (not np.array_equal(K,out[MAIN_ROLE][0])
            or not np.array_equal(d,out[MAIN_ROLE][1])
            or not meta["selected_from_explicit_main_camera_csv_row"]
            or not meta["source_csv_cross_group_filename_reference"]):
        raise ValueError("Main-camera source selection does not match declared row")
    return out


def compare_one_model_action(frames_by_subject, K, distortion, scale, action):
    """One fixed four-other-group train scale; all test predictions remain GT-free."""
    subjects = TARGET_A3_SUBJECTS if action == "A3" else (TARGET_GROUP,)
    participants=[]
    all_ref_project=[]
    for person in subjects:
        source = frames_by_subject[person][action]
        all_ref_project.extend(camera_projection_consistency(
            source["frames"], K, distortion))
        metric=test_recording(
            source["frames"], K, distortion, scale, action=action)
        participants.append(metric)
    if not all_ref_project:
        raise ValueError("No source-derived Vicon projection comparison")
    return {
        "n_people":len(participants),
        "valid_source_frames":sum(x["test_valid_six_joint_images"]
                                  for x in participants),
        "valid_PnP_frames":sum(x["PnP_location_predictions_with_positive_depth"]
                              for x in participants),
        "group_mean_root_error_pct_femur":round(
            mean(x["mean_root_error_pct_femur"] for x in participants),4),
        "group_mean_sixjoint_error_pct_femur":round(
            mean(x["mean_sixjoint_error_pct_femur"] for x in participants),4),
        "median_model_self_reprojection_px":round(
            median(x["median_self_reprojection_error_px"]
                   for x in participants),3),
        "source_xyz_to_source_xy_projection_median_px":round(
            float(np.median(all_ref_project)),3),
        "camera_reference_projection_p90_px":round(
            float(np.percentile(all_ref_project,90)),3),
        "private_person_summaries":participants,
    }


def reproduce_original_group_from_alt(alternate: dict, saved: dict):
    """Detect inference/subject/sample drift before replacing a frozen result."""
    for field in ("n_people","valid_source_frames","valid_PnP_frames",
                  "group_mean_root_error_pct_femur",
                  "group_mean_sixjoint_error_pct_femur",
                  "median_model_self_reprojection_px",
                  "camera_reference_projection_p90_px"):
        if alternate[field] != saved[field]:
            raise ValueError("Original S17 filename-based PnP reference not reproduced: "+field)
    return True


def five_group_sensitivity(original_private, new_main_role_by_model):
    """Replace only the two S17/C6 folds: never recompute other 4 from test data."""
    cells=[]
    rows=original_private["rows"]
    if (len(rows)!=20 or
        {(r["camera"],r["model"],r["heldout_group"]) for r in rows} !=
        {(c,m,g) for c in ("C1","C6") for m in VARIANTS for g in ANCHORS}):
        raise ValueError("Original five-group PnP report identities changed")
    for model in VARIANTS:
        for action in ("A3","A4"):
            existing = {
                group:next(r[action] for r in rows if
                           (r["camera"],r["model"],r["heldout_group"]) ==
                           (CAMERA,model,group))
                for group in ANCHORS
            }
            variant = dict(existing)
            variant[TARGET_GROUP] = new_main_role_by_model[model][action]
            by_condition={}
            for label,group_map in (
                ("previous_group_filename_rule",existing),
                ("source_explicit_main_camera_row",variant),
            ):
                by_condition[label]={
                    "group_equal_median_sixjoint_pct_femur":round(
                        median(group_map[g]["group_mean_sixjoint_error_pct_femur"]
                               for g in ANCHORS),4),
                    "group_equal_median_root_pct_femur":round(
                        median(group_map[g]["group_mean_root_error_pct_femur"]
                               for g in ANCHORS),4),
                    "max_group_reference_projection_p90_px":round(
                        max(group_map[g]["camera_reference_projection_p90_px"]
                            for g in ANCHORS),3),
                }
            cells.append({
                "camera":CAMERA,"model":model,"action":action,
                "original_independent_calibration_groups":len(ANCHORS),
                "S17_original_sixjoint_pct_femur":
                    existing[TARGET_GROUP]["group_mean_sixjoint_error_pct_femur"],
                "S17_source_role_sixjoint_pct_femur":
                    variant[TARGET_GROUP]["group_mean_sixjoint_error_pct_femur"],
                "S17_original_root_pct_femur":
                    existing[TARGET_GROUP]["group_mean_root_error_pct_femur"],
                "S17_source_role_root_pct_femur":
                    variant[TARGET_GROUP]["group_mean_root_error_pct_femur"],
                "S17_source_role_projection_p90_px":
                    variant[TARGET_GROUP]["camera_reference_projection_p90_px"],
                "aggregate_comparison":by_condition,
            })
    return cells


def run(root, models_dir, analysis_dir, private_path, aggregate_path):
    if (private_path==aggregate_path or
        any(p.exists() or p.parent!=analysis_dir for p in (private_path,aggregate_path))):
        raise ValueError("Output exists or is outside private research analysis dir")
    previous=json.loads((analysis_dir/
        "camera_PnP_LOCO_no_test_Vicon_private_20261119.json").read_text(
            encoding="utf-8"))
    protocol=json.loads((analysis_dir/
        "A3_26subject_D2_census_protocol_20261116.json").read_text(
            encoding="utf-8-sig"))
    if not previous.get("test_Vicon_centres_not_passed_to_prediction"):
        raise ValueError("Prior experiment did not keep Vicon out of predictor")
    candidates=source_role_candidates(root)
    models={}
    for variant in VARIANTS:
        path=models_dir/f"pose_landmarker_{variant}.task"
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=protocol["model_variants"][variant]["sha256"]:
            raise ValueError("Original model SHA changed")
        models[variant]=path
    private=[]
    new_group={}
    total_inferences=0
    total_detections=0
    for variant in VARIANTS:
        prior=next(r for r in previous["rows"] if (
            r["camera"],r["model"],r["heldout_group"]) ==
            (CAMERA,variant,TARGET_GROUP))
        scale=float(prior["training_source_units_per_model_m"])
        frame_map={}
        for person in TARGET_A3_SUBJECTS:
            frame_map[person]={"A3":read_inference(
                root,CAMERA,person,"A3",models[variant])}
            if person==TARGET_GROUP:
                frame_map[person]["A4"]=read_inference(
                    root,CAMERA,person,"A4",models[variant])
        for records in frame_map.values():
            for source in records.values():
                total_inferences+=source["attempted"]
                total_detections+=source["detected"]
                if source["attempted"]!=24 or not source["frozen_jpg_name_selection"]:
                    raise ValueError("Frozen original RGB source frames changed")
        candidate_results={}
        for role,(K,dist,meta) in candidates.items():
            candidate_results[role] = {
                action:compare_one_model_action(frame_map,K,dist,scale,action)
                for action in ("A3","A4")
            }
        for action in ("A3","A4"):
            reproduce_original_group_from_alt(
                candidate_results[PAIRED_ROLE][action],prior[action])
            for role in (MAIN_ROLE,PAIRED_ROLE):
                item=candidate_results[role][action]
                expected=(len(TARGET_A3_SUBJECTS) if action=="A3" else 1)
                if item["n_people"]!=expected:
                    raise ValueError("S17 testing population drift")
        new_group[variant]=candidate_results[MAIN_ROLE]
        private.append({
            "camera":CAMERA,"model":variant,
            "baseline_20261119_training_scale_reused":True,
            "training_scale_frozen_from_other_four_groups":True,
            "new_training_data_used":False,
            "csv_role_candidate_metadata":{
                role:inputs[2] for role,inputs in candidates.items()
            },
            "model_image_inferences":sum(
                source["attempted"] for rec in frame_map.values()
                for source in rec.values()),
            "candidate_results":candidate_results,
            "previous_alternate_csv_role_reproduced":True,
            "no_test_Vicon_bone_or_hip_used_for_PnP":True,
        })
    if total_inferences!=336 or total_detections<=0:
        raise ValueError("Expected exactly 336 inferences on 7 recordings×2 models")
    cells=five_group_sensitivity(previous,new_group)
    evidence_files=(
        "data/times/all_times_w_drops.csv",
        "data/stomp_sync_corrected_manual.json",
        "data/stomp_sync_refine_manual.json",
        "data/stomp_3d_manual.json",
    )
    timestamp_files={key:(root/key).is_file() for key in evidence_files}
    aggregate={
        "status":"source_role_calibration_provenance_sensitivity_complete",
        "source_csv_S17_C6_main_cam_role":"6",
        "source_csv_S17_C5_paired_cam_role":"5",
        "explicit_main_camera_row_references_other_named_group":True,
        "selected_before_evaluation_by_CSV_role_not_Vicon_error":True,
        "comparison_is_Vicon_derived_source_internal_consistency":True,
        "source_calibration_physical_authority_independently_verified":False,
        "camera_Vicon_hardware_sync_independently_verified":False,
        "original_time_artifacts_locally_present":timestamp_files,
        "reused_previous_frozen_four_other_groups":True,
        "S17_C6_original_paired_role_reproduced":True,
        "source_role_sensitivity_model_inferences":total_inferences,
        "source_role_sensitivity_pose_detections":total_detections,
        "original_five_calibration_groups":len(ANCHORS),
        "new_independent_camera_calibrations":0,
        "four_C6_camera_model_action_cells":cells,
        "physical_MPJPE_mm_validated":False,
        "raw_or_per_subject_data_exported":False,
    }
    details={"status":aggregate["status"],
             "original_pnp_full_five_group_report_unchanged":True,
             "main_camera_vs_paired_camera_role":private,
             "private_local_only":True,
             "no_per_frame_coordinates_or_images_saved":True}
    for name,obj in ((private_path,details),(aggregate_path,aggregate)):
        encoded=json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)
        with name.open("x",encoding="utf-8") as file:
            file.write(encoded+"\n")
    print(json.dumps(aggregate,ensure_ascii=False))
    return aggregate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--analysis-dir",type=Path,required=True)
    p.add_argument("--private-output",type=Path,required=True)
    p.add_argument("--aggregate-output",type=Path,required=True)
    args=p.parse_args(argv)
    run(args.root.resolve(),args.model_dir.resolve(),args.analysis_dir.resolve(),
        args.private_output.resolve(),args.aggregate_output.resolve())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
