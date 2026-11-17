"""Vicon reference-definition sensitivity: joint centres vs visible skin markers.

Does NOT assert marker points match MediaPipe landmarks or measure their error.
Reads only source WCS/camera metadata + ZIP member names, not RGB pixels.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import numpy as np

from .geometry import joint_angle
from .io import read_sample
from .benchmark import unique_sample_indices
from .position_holdout import ANCHORS, CAMERAS, strict_unique_pairs
from .depth_angle_decomposition import GROUPS

ACTIONS = ("A3", "A4")
ALTERNATIVES = ("knee_surface", "ankle_surface", "knee_and_ankle_surface")
BANDS = ("deep_lt120", "mid_120_150", "extended_ge150")


def _phase(deg: float) -> str:
    return BANDS[0] if deg < 120 else (BANDS[1] if deg < 150 else BANDS[2])


def named_marker_angle(record: dict, names: tuple[str, str, str]) -> float:
    ids = record["point_ids"]
    xyz = np.asarray(record["xyz"],float)
    if any(name not in ids for name in names):
        raise ValueError("Source lacks one of the declared marker labels")
    value = joint_angle(*(xyz[ids.index(name)] for name in names))
    if not math.isfinite(value):
        raise ValueError("Degenerate source marker geometry")
    return value


def reference_sensitivity(record: dict, side: str) -> dict:
    if side not in ("L", "R"):
        raise ValueError("Side must be left or right")
    hip, knee, ankle = (side+"HJC", side+"KJC", side+"AJC")
    sknee, sankle = side+"KNE", side+"ANK"
    center = named_marker_angle(record, (hip,knee,ankle))
    options = {
        "knee_surface":named_marker_angle(record,(hip,sknee,ankle)),
        "ankle_surface":named_marker_angle(record,(hip,knee,sankle)),
        "knee_and_ankle_surface":named_marker_angle(record,(hip,sknee,sankle)),
    }
    return {"band":_phase(center),**{
        name:angle-center for name,angle in options.items()
    }}


def _summarize(samples: list[dict]) -> dict:
    if not samples:
        return {"valid_knee_angles":0}
    metrics={"valid_knee_angles":len(samples)}
    for name in ALTERNATIVES:
        vals=[float(s[name]) for s in samples]
        metrics[name+"_mean_signed_deg"]=round(mean(vals),6)
        metrics[name+"_mean_absolute_deg"]=round(mean(abs(v) for v in vals),6)
        metrics[name+"_max_absolute_deg"]=round(max(abs(v) for v in vals),6)
    return metrics


def evaluate_recording(root: Path,subject:str,camera:str,action:str)->dict:
    paths,wcs,cam=read_sample(root,camera=camera,subject=subject,action=action,repeat="D2")
    if len(wcs)!=len(cam):
        raise ValueError("WCS/camera record counts differ")
    with ZipFile(paths["rgb"]) as archive:
        old,_=unique_sample_indices(cam,archive.namelist(),24)
        strict,qa=strict_unique_pairs(cam,archive.namelist(),count=24)
    if (len(old)!=24 or len(strict)!=24 or
        [name for name,_ in strict]!=[name for name,_ in old] or
        qa["ambiguous_or_unverifiable_jpg_groups"] or
        qa["selected_jpg_names_different_from_previous"]):
        raise ValueError("Source-valid 24-image sample not compatible with prior study")
    observations=[]
    for _,idx in strict:
        if (wcs[idx]["point_ids"]!=cam[idx]["point_ids"] or
            cam[idx]["dup_num"]!=cam[idx]["factor"]):
            raise ValueError("Invalid WCS or camera source association")
        for side in ("L","R"):
            observations.append(reference_sensitivity(wcs[idx],side))
    return {
        "subject":subject,
        "calibration_group":GROUPS[subject],
        "action":action,"camera":camera,
        "strict_unique_jpeg_samples":len(strict),
        "prior_invalid_midpoint_reference_pairs":qa["previous_index_selection_invalid_pairs"],
        "all":_summarize(observations),
        "by_phase":{phase:_summarize([o for o in observations if o["band"]==phase])
                    for phase in BANDS},
        "input_Vicon_source_has_skin_markers":True,
        "skin_markers_are_not_equivalent_to_MediaPipe_landmarks":True,
    }


def aggregate(rows:list[dict])->dict:
    for action in ACTIONS:
        target=set(GROUPS) if action=="A3" else set(ANCHORS)
        for cam in CAMERAS:
            subset=[r for r in rows if r["action"]==action and r["camera"]==cam]
            if len(subset)!=len(target) or {r["subject"] for r in subset}!=target:
                raise ValueError("Expected complete source-valid cohort and views")
    results=[]
    for action in ACTIONS:
        for cam in CAMERAS:
            for phase in ("all",*BANDS):
                selected=[r for r in rows if r["action"]==action and r["camera"]==cam]
                by_group=defaultdict(list)
                for r in selected:
                    metrics=r["all"] if phase=="all" else r["by_phase"][phase]
                    if metrics["valid_knee_angles"]:
                        by_group[r["calibration_group"]].append(metrics)
                result={"action":action,"camera":cam,"phase":phase,
                        "groups_with_data":len(by_group),
                        "participants_with_data":sum(len(x) for x in by_group.values()),
                        "valid_knee_angles":sum(z["valid_knee_angles"] for gs in by_group.values() for z in gs)}
                for alternative in ALTERNATIVES:
                    for measure in ("mean_absolute_deg","mean_signed_deg"):
                        key=alternative+"_"+measure
                        per_group=[mean(z[key] for z in arr) for arr in by_group.values()]
                        result[key+"_mean_of_group_means"]=(round(mean(per_group),4) if per_group else None)
                        result[key+"_median_group_mean"]=(round(median(per_group),4) if per_group else None)
                results.append(result)
    return {
        "status":"complete_exploratory_reference_marker_definition_sensitivity",
        "original_dataset":"SUMediPose source Vicon WCS with strict source-valid RGB matching",
        "independent_calibration_groups":5,
        "additional_independent_calibration_groups":0,
        "evaluated_recording_views":len(rows),
        "unique_jpeg_instances":sum(r["strict_unique_jpeg_samples"] for r in rows),
        "reference_side_angle_observations":sum(r["all"]["valid_knee_angles"] for r in rows),
        "model_inferences":0,
        "alternatives":{
            "knee_surface":"HJC + surface KNE + joint-centre AJC",
            "ankle_surface":"HJC + joint-centre KJC + surface ANK",
            "knee_and_ankle_surface":"HJC + surface KNE + surface ANK",
        },
        "results":results,
        "limitations":{
            "skin_markers_are_not_clinically_equivalent_to_MediaPipe_landmarks":True,
            "changing_the_reference_does_not_measure_model_joint_location_accuracy":True,
            "camera_Vicon_physical_exposure_timestamp_not_verified":True,
            "model_landmark_vs_Vicon_anatomical_definition_not_harmonized":True,
            "not_a_clinically_validated_angular_threshold":True,
            "source_skin_markers_can_move_with_skin":True,
            "only_nonidentifying_group_level_results_shareable":True,
        },
    }


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--analysis-dir",type=Path,required=True)
    ap.add_argument("--private-output",type=Path,required=True)
    ap.add_argument("--aggregate-output",type=Path,required=True)
    args=ap.parse_args(argv)
    repo=Path(__file__).resolve().parents[2]
    folder=args.analysis_dir.resolve()
    outputs=(args.private_output.resolve(),args.aggregate_output.resolve())
    if (outputs[0]==outputs[1] or any(
        p.exists() or p.parent!=folder or p==repo or repo in p.parents
        for p in outputs
    )):
        ap.error("Output only to new private analysis files outside code repository")
    rows=[evaluate_recording(args.root,sub,c,act) for act in ACTIONS
          for sub in (tuple(GROUPS) if act=="A3" else ANCHORS) for c in CAMERAS]
    overview=aggregate(rows)
    private={
        "status":overview["status"],
        "private_participant_rows":rows,
        "no_raw_RGB_video_XYZ_or_per_frame_landmarks_saved":True,
    }
    for path,content in zip(outputs,(private,overview)):
        with path.open("x",encoding="utf8") as f:
            json.dump(content,f,ensure_ascii=False,indent=2)
            f.write("\n")
    print(json.dumps({
        "status":overview["status"],
        "views":overview["evaluated_recording_views"],
        "jpeg_instances":overview["unique_jpeg_instances"],
        "reference_side_angles":overview["reference_side_angle_observations"],
        "all_phase":[{k:row[k] for k in ("action","camera","valid_knee_angles",
            "knee_surface_mean_absolute_deg_mean_of_group_means",
            "ankle_surface_mean_absolute_deg_mean_of_group_means",
            "knee_and_ankle_surface_mean_absolute_deg_mean_of_group_means")}
            for row in overview["results"] if row["phase"]=="all"],
    }))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
