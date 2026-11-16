"""Local-only MediaPipe 2D/3D angular feasibility baseline.

Both 2D and 3D outputs come from ONE network, not two independent models.
Reference 2D angle is projected Vicon geometry and 3D angle is Vicon WCS.
They have different geometric definitions; never rank 2D vs 3D by raw MAE.
"""
from __future__ import annotations
from collections import defaultdict
import io
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from .geometry import joint_angle
from .io import read_sample

JOINT_NAMES = {
    "left": {"reference": ("LHJC", "LKJC", "LAJC"), "pred": (23, 25, 27)},
    "right": {"reference": ("RHJC", "RKJC", "RAJC"), "pred": (24, 26, 28)},
}


def unique_sample_indices(records, image_members, max_frames=40):
    """Evenly sample available unique RGB names and one coordinate index each."""
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[Path(str(r.get("path", ""))).name].append(i)
    available = set(image_members)
    entries = [(name, indices[len(indices)//2])
               for name, indices in groups.items() if name in available]
    picks = np.linspace(0, len(entries)-1, num=min(max_frames, len(entries)), dtype=int) if entries else []
    return [entries[int(i)] for i in picks], len(groups)-len(entries)


def _angles_from_named(record, names, dimension):
    ids = record["point_ids"]
    xyz = np.asarray(record[dimension], dtype=float)
    return joint_angle(*(xyz[ids.index(name)] for name in names))


def summarize_errors(errors, attempted):
    valid = [float(e) for e in errors if math.isfinite(float(e))]
    if not valid:
        return {"valid":0, "attempted":attempted, "MAE_degrees":None,
                "RMSE_degrees":None,"median_absolute_error_degrees":None}
    a = np.asarray(valid, dtype=float)
    return {"valid":int(len(a)),"attempted":int(attempted),
            "MAE_degrees":round(float(a.mean()),4),
            "RMSE_degrees":round(float(np.sqrt(np.mean(a*a))),4),
            "median_absolute_error_degrees":round(float(np.median(a)),4),
            "p90_abs_error_degrees":round(float(np.percentile(a,90)),4),
            "fraction_abs_error_over_20_degrees":round(float(np.mean(a>20)),4)}


def evaluate_pose_model(root, model_asset_path, camera="C1", subject="S6",
                       repeat="D2", max_frames=40, min_visibility=.5):
    """Run actual RGB inference; return aggregate errors, no raw images or keypoints."""
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be in [0,1]")
    import mediapipe as mp
    import hashlib
    import cv2
    model_asset_path = Path(model_asset_path)
    if not model_asset_path.is_file():
        raise FileNotFoundError(f"Model asset missing: {model_asset_path}")
    paths, world, cam = read_sample(root,camera=camera,subject=subject,
                                   action="A3",repeat=repeat)
    if len(world)!=len(cam):
        raise ValueError("WCS and camera records must be index aligned")
    errors = {"image_2d":{"left":[],"right":[]},"world_3d":{"left":[],"right":[]}}
    detections = 0
    try:
        option = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_asset_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        with ZipFile(paths["rgb"]) as archive:
            picks, missing = unique_sample_indices(cam, archive.namelist(), max_frames)
            with mp.tasks.vision.PoseLandmarker.create_from_options(option) as task:
                for name,idx in picks:
                    jpg=cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),cv2.IMREAD_COLOR)
                    if jpg is None:
                        continue
                    h,w=jpg.shape[:2]
                    rgb=np.ascontiguousarray(cv2.cvtColor(jpg,cv2.COLOR_BGR2RGB))
                    detection=task.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                    if not detection.pose_landmarks or not detection.pose_world_landmarks:
                        continue
                    detections+=1
                    xy=detection.pose_landmarks[0]
                    xyz=detection.pose_world_landmarks[0]
                    for side, maps in JOINT_NAMES.items():
                        indices=maps["pred"]
                        if any(xy[j].visibility<min_visibility or xyz[j].visibility<min_visibility for j in indices):
                            continue
                        pred2=[[xy[j].x*w,xy[j].y*h] for j in indices]
                        pred3=[[xyz[j].x,xyz[j].y,xyz[j].z] for j in indices]
                        try:
                            gt2=_angles_from_named(cam[idx],maps["reference"],"xy")
                            gt3=_angles_from_named(world[idx],maps["reference"],"xyz")
                            pred2deg=joint_angle(*pred2)
                            pred3deg=joint_angle(*pred3)
                            if np.isfinite(gt2) and np.isfinite(pred2deg):
                                errors["image_2d"][side].append(abs(pred2deg-gt2))
                            if np.isfinite(gt3) and np.isfinite(pred3deg):
                                errors["world_3d"][side].append(abs(pred3deg-gt3))
                        except (ValueError,IndexError,KeyError,TypeError):
                            continue
    except Exception:
        raise
    summary={modality:{side:summarize_errors(errors[modality][side],len(picks))
                 for side in ("left","right")} for modality in errors}
    with model_asset_path.open("rb") as fh:
        model_hash = hashlib.sha256(fh.read()).hexdigest()
    return {
        "sample":f"{camera}{subject}A3{repeat}",
        "model":"Google MediaPipe Pose Landmarker (one model, two output spaces)",
        "model_variant":model_asset_path.stem.replace("pose_landmarker_",""),
        "model_asset_sha256":model_hash,
        "mediapipe_version":mp.__version__,
        "inference_mode":"IMAGE",
        "max_unique_images_requested":max_frames,
        "unique_images_sampled":len(picks),
        "model_detected_poses":detections,
        "rgb_unavailable_references_masked":missing,
        "minimum_landmark_visibility":min_visibility,
        "results":summary,
        "2d_reference":"included 2D knee angle from Vicon-projected joint centres on image",
        "3d_reference":"included 3D knee angle from Vicon WCS joint centres",
        "different_joint_definitions_and_projection_geometry":True,
        "2d_vs_3d_absolute_errors_not_directly_rankable":True,
        "not_independent_two_model_comparison":True,
        "reference_joint_centres_and_predicted_landmark_anatomy_not_identical":True,
        "medical_accuracy_validated":False,
        "raw_images_or_per_frame_poses_saved":False,
    }
