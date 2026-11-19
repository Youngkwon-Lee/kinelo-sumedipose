"""Experimental image-only test-time 3D root localization via calibrated PnP.

Fitted SCALE comes only from four OTHER calibration groups' A3 first-12
training frames. At test time, predict_camera_points uses ONLY MediaPipe
2D pixels, 3D model landmarks, frozen training scale, and the target
camera's source-provided intrinsic matrix/distortion coefficients.

Vicon ground-truth XYZ enters ONLY evaluate_prediction AFTER inference,
never model-world calibration or PnP fit on the held-out group.
The physical intrinsics, source camera->Vicon transform, anatomical point
definitions, and exposure timestamps are not independently validated.
These numbers are exploratory source-camera-XYZ discrepancies, NOT
verified clinical MPJPE(mm), autonomous recovery of Vicon WCS, or
evidence of no held-out camera calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import numpy as np

from .io import read_sample
from .position_holdout import (
    ANCHORS, CAMERAS, VARIANTS, GT_JOINTS, MODEL_JOINTS,
    strict_unique_pairs,
)

ALL26 = tuple(f"S{i}" for i in range(1, 29) if i not in (5, 10))
PAIR_BY_CAMERA = {"C1": ("G12", "1"), "C6": ("G65", "6")}
TRAIN_SLOTS = 12
IMAGES_PER_RECORDING = 24
MIN_TRAIN_SIX_JOINT_FRAMES = 6
MIN_HELDOUT_VALID_FRAMES = 6
REFERENCE_PROJECTION_WARNING_P90_PX = 20.0


def calibration_intrinsics(root: Path, group: str, camera: str):
    """Use the source CSV's preselected main-camera calibration, no GT-based pick.

    In G12, camera 1 is stereo main (the CSV row may be for cam=2).
    In G65, camera 6 is stereo main. Both use mtx_A/dist_A.
    Reject any CSV path whose matrix filename belongs to another group;
    the released S17/C6 CSV includes a cross-group path for S23.
    """
    if group not in ANCHORS or camera not in PAIR_BY_CAMERA:
        raise ValueError("Unexpected source group or camera")
    pair, main_id = PAIR_BY_CAMERA[camera]
    source = root / "calibration" / "external_calibration.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = {r["path"] for r in rows
                  if r["subject"] == group
                  and r["main"] == main_id
                  and f"_{pair}_" in r["path"]
                  and Path(r["path"]).name.startswith(f"S{group}_{pair}_")
                  and r["main_link"].lower() in ("true", "false")}
    if len(candidates) != 1:
        raise ValueError("Camera intrinsics cannot be identified uniquely from calibration CSV")
    rel = next(iter(candidates))
    path = (root / rel).resolve()
    if (root.resolve() not in path.parents or
            path.suffix.lower() != ".npz" or not path.is_file()):
        raise ValueError("Unexpected source calibration matrix path")
    with np.load(path, allow_pickle=False) as data:
        K = np.asarray(data["mtx_A"], dtype=float)
        distortion = np.asarray(data["dist_A"], dtype=float)
    if (K.shape != (3, 3) or distortion.size not in (4, 5, 8, 12, 14) or
            not np.isfinite(K).all() or not np.isfinite(distortion).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[2, 2]-1) > 1e-8):
        raise ValueError("Malformed source camera intrinsics")
    return K, distortion


def training_group_scale(frames):
    """Group median GT-CCS/metre femur ratio, TRAINING ONLY."""
    vals = []
    for item in frames:
        if item["slot"] >= TRAIN_SLOTS:
            raise ValueError("Test frame included in training calibration")
        m = item["model_3d"]
        ref = item["reference_camera_xyz"]
        lm = mean((float(np.linalg.norm(m[i]-m[j])) for i,j
                   in ((0,1),(3,4))))
        lg = mean((float(np.linalg.norm(ref[i]-ref[j])) for i,j
                   in ((0,1),(3,4))))
        if math.isfinite(lm) and math.isfinite(lg) and lm > 1e-8 and lg > 1e-8:
            vals.append(lg/lm)
    if len(vals) < MIN_TRAIN_SIX_JOINT_FRAMES:
        raise ValueError("Not enough training six-joint reference frames")
    return float(median(vals))


def fixed_scale_excluding_group(anchor_source, heldout: str):
    """Train one scale on four OTHER independent calibration-group anchors."""
    if heldout not in ANCHORS or set(anchor_source) != set(ANCHORS):
        raise ValueError("LOCO frame input missing original calibration groups")
    learned = {}
    for group in ANCHORS:
        if group == heldout:
            continue
        samples = anchor_source[group]
        valid_train = [r for r in samples if r["slot"] < TRAIN_SLOTS]
        learned[group] = training_group_scale(valid_train)
    if len(learned) != 4:
        raise ValueError("LOCO scale did not exclude exactly one group")
    return float(median(learned.values()))


def predict_camera_points(model_xyz, image_xy, K, distortion, source_units_per_m):
    """PnP image-only inference: NO Vicon/ground truth arg or access.

    Image 2D points and 3D model landmarks are outputs from the same RGB
    MediaPipe inference. Source camera K is pre-provided for this rig.
    No test Vicon location, orientation or bone length is supplied.
    """
    import cv2
    xyz = np.asarray(model_xyz, dtype=float)
    xy = np.asarray(image_xy, dtype=float)
    K = np.asarray(K, dtype=float)
    distortion = np.asarray(distortion, dtype=float)
    scale = float(source_units_per_m)
    if (xyz.shape != (6, 3) or xy.shape != (6, 2) or
            K.shape != (3, 3) or not np.isfinite(xyz).all()
            or not np.isfinite(xy).all() or not np.isfinite(K).all()
            or not np.isfinite(distortion).all() or
            not math.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid model-only PnP input or frozen training scale")
    object_pts = np.ascontiguousarray(xyz * scale, dtype=np.float64)
    image_pts = np.ascontiguousarray(xy, dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_pts, image_pts, K, distortion,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        pred = (object_pts @ R.T) + tvec.reshape(1, 3)
        projected, _ = cv2.projectPoints(
            object_pts, rvec, tvec, K, distortion)
    except (cv2.error, ValueError):
        return None
    if (not np.isfinite(pred).all() or
            np.min(pred[:, 2]) <= 0):
        return None
    reproj = np.linalg.norm(projected.reshape(-1, 2) - xy, axis=1)
    return {
        "predicted_xyz_camera_source_units": pred,
        "mean_model_self_reprojection_error_px": float(reproj.mean()),
        "pelvis_predicted_xyz_camera_source_units": pred[[0,3]].mean(axis=0),
        "test_ground_truth_queried_for_prediction": False,
    }


def evaluate_prediction(prediction, reference_camera_xyz):
    """Test Vicon GT used ONLY here as an evaluation label, never PnP input."""
    gt = np.asarray(reference_camera_xyz, float)
    pred = prediction["predicted_xyz_camera_source_units"]
    if gt.shape != (6,3) or not np.isfinite(gt).all():
        raise ValueError("Invalid reference evaluation coordinates")
    femur = mean((float(np.linalg.norm(gt[i]-gt[j]))
                  for i,j in ((0,1),(3,4))))
    if not math.isfinite(femur) or femur <= 1e-8:
        raise ValueError("Invalid source reference segment length")
    joint_distance = float(np.linalg.norm(pred-gt,axis=1).mean())
    gt_pelvis = gt[[0,3]].mean(axis=0)
    root_distance = float(np.linalg.norm(
        prediction["pelvis_predicted_xyz_camera_source_units"]-gt_pelvis))
    return {
        "full_joint_position_discrepancy_pct_reference_femur":
            100*joint_distance/femur,
        "root_position_discrepancy_pct_reference_femur":
            100*root_distance/femur,
        "camera_native_mean_joint_distance": joint_distance,
        "Vicon_GT_not_used_in_location_prediction": True,
        "validated_physical_MPJPE_mm": False,
    }


def read_inference(root: Path, camera: str, subject: str, action: str,
                   task_path: Path, visibility=0.5):
    """Keep all source/per-frame data only in runtime memory."""
    import cv2
    import mediapipe as mp

    paths,wcs,cam_records = read_sample(
        root, camera=camera, subject=subject, action=action, repeat="D2")
    if len(wcs)!=len(cam_records):
        raise ValueError("Source reference record lengths differ")
    with ZipFile(paths["rgb"]) as archive:
        picks, source_qa = strict_unique_pairs(
            cam_records, archive.namelist(), IMAGES_PER_RECORDING)
        if source_qa["selected_jpg_names_different_from_previous"]:
            raise ValueError("Original frozen RGB selection changed")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1)
        frames,detected = [],0
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for slot,(name,idx) in enumerate(picks):
                jpg=cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),cv2.IMREAD_COLOR)
                if jpg is None:continue
                h,w=jpg.shape[:2]
                rgb=np.ascontiguousarray(cv2.cvtColor(jpg,cv2.COLOR_BGR2RGB))
                result=detector.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB,data=rgb))
                if not result.pose_landmarks or not result.pose_world_landmarks:
                    continue
                detected+=1
                world=result.pose_world_landmarks[0]
                image=result.pose_landmarks[0]
                if any(world[j].visibility < visibility or
                       image[j].visibility < visibility
                       for j in MODEL_JOINTS):
                    continue
                indices=[cam_records[idx]["point_ids"].index(v)
                         for v in GT_JOINTS]
                xyz=np.asarray(cam_records[idx]["xyz"],dtype=float)[indices]
                ref_xy=np.asarray(cam_records[idx]["xy"],dtype=float)[indices]
                mp_xyz=np.asarray(
                    [[world[j].x,world[j].y,world[j].z] for j in MODEL_JOINTS],
                    dtype=float)
                mp_xy=np.asarray(
                    [[image[j].x*w,image[j].y*h] for j in MODEL_JOINTS],
                    dtype=float)
                if all(np.isfinite(x).all()
                       for x in (xyz,ref_xy,mp_xyz,mp_xy)):
                    frames.append({
                        "slot":slot, "model_3d":mp_xyz,"model_xy":mp_xy,
                        "reference_camera_xyz":xyz,
                        "reference_camera_xy":ref_xy,
                    })
    return {
        "frames":frames,
        "attempted":len(picks),
        "detected":detected,
        "valid_six_joint":len(frames),
        "frozen_jpg_name_selection":True,
        "missing_source_reference_groups":source_qa["missing_original_jpg_groups"],
    }


def camera_projection_consistency(frames, K, distortion):
    """Reference integrity check; never used to choose training scale or PnP."""
    import cv2
    errors=[]
    for item in frames:
        if item["slot"] < 0:
            raise ValueError("Invalid source frame slot")
        xy,_=cv2.projectPoints(
            item["reference_camera_xyz"],
            np.zeros(3),np.zeros(3),K,distortion)
        e=np.linalg.norm(xy.reshape(-1,2)-item["reference_camera_xy"],axis=1)
        errors.extend(e.tolist())
    return errors


def test_recording(frames, K, distortion, scale, *, action):
    """All predictions use model pixel+world only; reference used AFTER PnP."""
    if action not in ("A3","A4"):
        raise ValueError("Unsupported evaluation action")
    candidates=[f for f in frames if
                action=="A4" or f["slot"]>=TRAIN_SLOTS]
    if len(candidates)<MIN_HELDOUT_VALID_FRAMES:
        raise ValueError("Insufficient six-joint holdout frames")
    reprojs,root_dist,joint_dist=[],[],[]
    n_failure=0
    for f in candidates:
        prediction=predict_camera_points(
            f["model_3d"],f["model_xy"],K,distortion,scale)
        if prediction is None:
            n_failure+=1;continue
        # This is the ONLY test-time use of Vicon GT.
        value=evaluate_prediction(prediction,f["reference_camera_xyz"])
        reprojs.append(prediction["mean_model_self_reprojection_error_px"])
        root_dist.append(value["root_position_discrepancy_pct_reference_femur"])
        joint_dist.append(value["full_joint_position_discrepancy_pct_reference_femur"])
    if len(joint_dist)<MIN_HELDOUT_VALID_FRAMES:
        raise ValueError("Too few valid model-only PnP predictions in test fold")
    return {
        "test_valid_six_joint_images":len(candidates),
        "PnP_location_predictions_with_positive_depth":len(joint_dist),
        "PnP_prediction_failures":n_failure,
        "mean_root_error_pct_femur":round(float(mean(root_dist)),4),
        "mean_sixjoint_error_pct_femur":round(float(mean(joint_dist)),4),
        "median_self_reprojection_error_px":round(float(median(reprojs)),3),
        "fraction_self_reprojection_gt20px":round(float(np.mean(np.array(reprojs)>20)),4),
        "test_Vicon_used_in_prediction":False,
        "metric_MPJPE_mm_validated":False,
    }


def aggregate_fold_rows(rows):
    """Five held-out independent historical calibration-group folds."""
    expected={(c,m,g) for c in CAMERAS for m in VARIANTS for g in ANCHORS}
    if len(rows)!=20 or {(r["camera"],r["model"],r["heldout_group"])
                           for r in rows}!=expected:
        raise ValueError("Require exactly five independent calibration folds per camera/model")
    cells=[]
    for camera in CAMERAS:
        for model in VARIANTS:
            folds=[r for r in rows if r["camera"]==camera and r["model"]==model]
            for act in ("A3","A4"):
                per_group=[r[act] for r in folds]
                vals=[x["group_mean_sixjoint_error_pct_femur"] for x in per_group]
                roots=[x["group_mean_root_error_pct_femur"] for x in per_group]
                cells.append({
                    "camera":camera,"model":model,"action":act,
                    "n_historical_calibration_groups":5,
                    "n_heldout_subjects":sum(x["n_people"] for x in per_group),
                    "positive_depth_PnP_test_frames":sum(x["valid_PnP_frames"] for x in per_group),
                    "test_valid_frames_before_PnP":sum(x["valid_source_frames"] for x in per_group),
                    "median_equal_group_sixjoint_error_pct_femur":round(float(median(vals)),4),
                    "median_equal_group_root_error_pct_femur":round(float(median(roots)),4),
                    "min_max_group_mean_sixjoint_pct_femur":[round(min(vals),4),round(max(vals),4)],
                    "median_model_self_reprojection_px":round(
                        float(median(x["median_model_self_reprojection_px"] for x in per_group)),3),
                    "max_camera_reference_projection_p90_px":round(
                        max(x["camera_reference_projection_p90_px"] for x in per_group),3),
                    "camera_source_intrinsics_evidence_warning":any(
                        x["camera_reference_projection_p90_px"] >
                        REFERENCE_PROJECTION_WARNING_P90_PX for x in per_group),
                })
    return {
        "status":"exploratory_model_only_camera_PnP_localization",
        "training":"Other 4 groups' A3 first 12 source-valid images for fixed femur scale",
        "prediction_inputs":"MediaPipe 2D/3D RGB output, source camera intrinsics and training scale; no test Vicon",
        "reference":"Vicon-derived camera XYZ, used only after PnP to score",
        "camera_groups_available":5,
        "new_independent_camera_calibrations":0,
        "A3_heldout_subjects":26,
        "A4_heldout_anchors":5,
        "folds":20,
        "cells":cells,
        "no_test_Vicon_root_or_bone_length_used_for_prediction":True,
        "target_group_camera_intrinsics_were_provided":True,
        "camera_intrinsics_not_independently_verified_as_metric":True,
        "Vicon_camera_source_capture_timestamp_independently_verified":False,
        "MediaPipe_and_Vicon_anatomical_joint_definitions_harmonized":False,
        "absolute_MPJPE_mm_clinically_validated":False,
        "no_raw_images_predicted_keypoints_or_subject_rows_in_public_research_report":True,
    }


def run(root, models_dir, analysis_dir, detail_path, aggregate_path):
    import cv2
    if (detail_path==aggregate_path or
        any(p.exists() or p.parent!=analysis_dir for p in (detail_path,aggregate_path))):
        raise ValueError("Private outputs must be distinct new files under approved analysis folder")
    repo=Path(__file__).resolve().parents[2]
    if any(p==repo or repo in p.parents for p in (detail_path,aggregate_path)):
        raise ValueError("Refuse private data output inside public code checkout")
    census=json.loads((analysis_dir/"A3_26subject_D2_census_protocol_20261116.json").read_text(encoding="utf-8-sig"))
    subject_groups=json.loads((root/"calibration/external/subject_map.json").read_text(encoding="utf-8-sig"))
    if set(subject_groups[s] for s in ALL26)!=set(ANCHORS):
        raise ValueError("Original calibration group mapping changed")
    group_members={g:[s for s in ALL26 if subject_groups[s]==g] for g in ANCHORS}
    if {g:len(v) for g,v in group_members.items()}!=census["calibration_group_sizes"]:
        raise ValueError("Original five group sizes changed")
    model_paths={}
    for m in VARIANTS:
        path=models_dir/f"pose_landmarker_{m}.task"
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=census["model_variants"][m]["sha256"]:
            raise ValueError("Frozen model asset SHA256 changed")
        model_paths[m]=path
    private_rows=[]
    attempt_images=0;detected_poses=0
    for camera in CAMERAS:
        Ks={g:calibration_intrinsics(root,g,camera) for g in ANCHORS}
        for model in VARIANTS:
            source={}
            for s in ALL26:
                a3=read_inference(root,camera,s,"A3",model_paths[model])
                source[s]={"A3":a3}
                attempt_images+=a3["attempted"];detected_poses+=a3["detected"]
                if a3["attempted"]!=24:
                    raise ValueError("Frozen A3 JPG count changed")
                if s in ANCHORS:
                    a4=read_inference(root,camera,s,"A4",model_paths[model])
                    if a4["attempted"]!=24:
                        raise ValueError("Frozen A4 JPG count changed")
                    source[s]["A4"]=a4
                    attempt_images+=a4["attempted"];detected_poses+=a4["detected"]
            for group in ANCHORS:
                scale=fixed_scale_excluding_group({
                    g:source[g]["A3"]["frames"] for g in ANCHORS}, group)
                fold={"heldout_group":group,"camera":camera,"model":model,
                    "calibration_group_used_only_for_target_intrinsics":True,
                    "scale_fitted_only_on_other_four_groups":True,
                    "training_source_units_per_model_m":round(scale,6),
                    "A3":None,"A4":None}
                for action in ("A3","A4"):
                    people=group_members[group] if action=="A3" else [group]
                    result=[]
                    quality_errors=[]
                    for person in people:
                        recording=source[person][action]
                        K,dist=Ks[group]
                        errors=camera_projection_consistency(recording["frames"],K,dist)
                        quality_errors.extend(errors)
                        scoring=test_recording(recording["frames"],K,dist,scale,action=action)
                        scoring["subject"]=person
                        result.append(scoring)
                    if not quality_errors:
                        raise ValueError("Missing reference image projection integrity observations")
                    fold[action]={
                        "n_people":len(result),
                        "valid_source_frames":sum(z["test_valid_six_joint_images"] for z in result),
                        "valid_PnP_frames":sum(z["PnP_location_predictions_with_positive_depth"] for z in result),
                        "group_mean_root_error_pct_femur":round(float(mean(z["mean_root_error_pct_femur"] for z in result)),4),
                        "group_mean_sixjoint_error_pct_femur":round(float(mean(z["mean_sixjoint_error_pct_femur"] for z in result)),4),
                        "median_model_self_reprojection_px":round(float(median(z["median_self_reprojection_error_px"] for z in result)),3),
                        "camera_reference_projection_p90_px":round(float(np.percentile(quality_errors,90)),3),
                        "private_person_summaries":result,
                    }
                private_rows.append(fold)
            print(json.dumps({"progress":"completed_camera_model","camera":camera,
                              "model":model,"completed_LOCO_folds":len(private_rows)}),
                  flush=True)
    if attempt_images!=2976 or not 0<detected_poses<=attempt_images:
        raise ValueError("Unexpected total MediaPipe image inferences or detections")
    overview=aggregate_fold_rows(private_rows)
    overview["model_image_inferences"]=attempt_images
    overview["pose_detections"]=detected_poses
    overview["model_task_sha256_verified"]=True
    private={
        "status":overview["status"],"rows":private_rows,
        "original_model_SHA256":{m:hashlib.sha256(v.read_bytes()).hexdigest()
                                 for m,v in model_paths.items()},
        "test_Vicon_centres_not_passed_to_prediction":True,
        "original_RGB_and_per_frame_keypoints_not_saved":True,
        "private_local_only":True,
    }
    # Failure-closed strict JSON with real newline; do not overwrite originals.
    for file,obj in ((detail_path,private),(aggregate_path,overview)):
        encoded=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
        with file.open("x",encoding="utf-8") as f:
            f.write(encoded+"\n")
    print(json.dumps({k:overview[k] for k in (
        "status","A3_heldout_subjects","A4_heldout_anchors",
        "model_image_inferences","pose_detections","cells")},ensure_ascii=False),
        flush=True)
    return overview


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--models-dir",type=Path,required=True)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--private-output",type=Path,required=True)
    parser.add_argument("--aggregate-output",type=Path,required=True)
    args=parser.parse_args(argv)
    run(args.root.resolve(),args.models_dir.resolve(),args.analysis_dir.resolve(),
        args.private_output.resolve(),args.aggregate_output.resolve())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
