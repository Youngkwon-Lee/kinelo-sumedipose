"""Leave-one-camera-calibration-group-out MediaPipe 3D shape transfer.

Fit model→Vicon root-relative scale/orientation on A3 first-12 frames of
anchors in the OTHER four historical calibration groups only. Test on the
held-out group's 26 A3 participants (last 12) and its anchor's A4 (all 24).

Test Vicon GT hip centering is an evaluation oracle; no world-space MPJPE,
anatomical landmark equivalence, shared-clock evidence, or clinical claim.
All individual model predictions and sources stay on approved private storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

import numpy as np

from .position_holdout import (
    ANCHORS, CAMERAS, VARIANTS, MIN_VALID_PER_SET,
    read_pose_frames, heldout_discrepancy,
)
from .mediapipe_parity import fit_parity_similarity

TOTAL_SUBJECTS = tuple(f"S{i}" for i in range(1, 29) if i not in (5, 10))
OUTCOMES = ("A3_heldout", "A4_cross_action")
PARITIES = ("proper", "reflected")
DETERMINANT = {"proper": 1, "reflected": -1}
TRAIN_LIMIT = 12
MIN_PER_GROUP = 6
SAMPLE_COUNT = 24


def validate_groups(subject_to_group, census):
    """All 26 persons mapped to five declared calibration groups (not n=26)."""
    if (len(TOTAL_SUBJECTS) != 26
        or set(subject_to_group[s] for s in TOTAL_SUBJECTS) != set(ANCHORS)
        or set(census["missing_subjects"]) != {"S5", "S10"}):
        raise ValueError("Original 26-person A3 source cohort/group map changed")
    actual = {g: sum(subject_to_group[s] == g for s in TOTAL_SUBJECTS)
              for g in ANCHORS}
    if actual != census["calibration_group_sizes"]:
        raise ValueError("Unexpected five camera-calibration group sizes")
    return {g: tuple(s for s in TOTAL_SUBJECTS
                     if subject_to_group[s] == g)
            for g in ANCHORS}


def _only_a3_first12(frames):
    return [r for r in frames if r[0] < TRAIN_LIMIT]


def _only_a3_last12(frames):
    return [r for r in frames if r[0] >= TRAIN_LIMIT]


def training_only_fits(train_frames):
    """Select parity on TRAIN frames only; never test set scores."""
    if len(train_frames) < 4*MIN_PER_GROUP:
        raise ValueError("Missing Vicon-labelled train frames from other groups")
    x = np.stack([f[1] for f in train_frames])
    y = np.stack([f[2] for f in train_frames])
    fits = {}
    for name, det in DETERMINANT.items():
        rot, scale = fit_parity_similarity(x, y, determinant=det)
        fit_score = heldout_discrepancy(train_frames, rot, scale)
        fits[name] = {
            "rotation": rot, "scale": scale,
            "training_pct_reference_femur":
                fit_score["mean_discrepancy_pct_reference_femur_length"],
        }
    # Frozen, fully predeclared choice from training only; tie prefers proper.
    chosen = min(PARITIES, key=lambda k: (
        fits[k]["training_pct_reference_femur"], k != "proper"
    ))
    return fits, chosen


def evaluate_excluded_group(by_subject, group_members, heldout_group):
    """Train exactly other four anchor subjects and test only the unseen group."""
    anchors_used = tuple(g for g in ANCHORS if g != heldout_group)
    if len(anchors_used) != 4 or heldout_group in anchors_used:
        raise ValueError("LOCO split training leaked heldout calibration group")
    train = []
    for g in anchors_used:
        selected = _only_a3_first12(by_subject[g]["A3"][0])
        if len(selected) < MIN_PER_GROUP:
            raise ValueError("Training calibration anchor lacks >=6 valid A3 frames")
        train.extend(selected)
    fits, chosen = training_only_fits(train)
    entries = []
    for subject in group_members:
        if subject in anchors_used:
            raise ValueError("Training person found in heldout calibration group")
        a3 = _only_a3_last12(by_subject[subject]["A3"][0])
        if len(a3) < MIN_VALID_PER_SET:
            raise ValueError("Heldout participant has too few valid A3 frames")
        result = {
            "subject": subject,
            "evaluation": "A3_heldout",
            "valid_test_frames": len(a3),
            "scores": {
                kind: heldout_discrepancy(
                    a3, fit["rotation"], fit["scale"])[
                        "mean_discrepancy_pct_reference_femur_length"]
                for kind, fit in fits.items()
            },
        }
        result["selected_training_only"] = result["scores"][chosen]
        entries.append(result)
        if subject == heldout_group:
            a4 = by_subject[subject]["A4"][0]
            if len(a4) < MIN_VALID_PER_SET:
                raise ValueError("Heldout A4 anchor has too few valid frames")
            a4_result = {
                "subject": subject,
                "evaluation": "A4_cross_action",
                "valid_test_frames": len(a4),
                "scores": {
                    kind: heldout_discrepancy(
                        a4, fit["rotation"], fit["scale"])[
                            "mean_discrepancy_pct_reference_femur_length"]
                    for kind, fit in fits.items()
                },
            }
            a4_result["selected_training_only"] = a4_result["scores"][chosen]
            entries.append(a4_result)
    return {
        "heldout_group": heldout_group,
        "train_anchor_groups": anchors_used,
        "train_valid_frames": len(train),
        "train_fit_pct_femur": {
            k: v["training_pct_reference_femur"] for k, v in fits.items()
        },
        "parity_chosen_from_train_only": chosen,
        "private_test_participant_rows": entries,
        "test_group_was_never_used_for_fit": True,
        "model_weights_frozen": True,
    }


def summarize(rows, group_to_members, within_person_reference=None):
    """Five independent calibration groups, 26 A3 subjects and 5 A4 anchors."""
    expected = {(c, m, g) for c in CAMERAS for m in VARIANTS
                for g in ANCHORS}
    if len(rows) != 20 or {(r["camera"],r["model"],r["heldout_group"])
                           for r in rows} != expected:
        raise ValueError("Missing or duplicate LOCO model/view/calibration fold")
    for r in rows:
        group = r["heldout_group"]
        if (set(r["train_anchor_groups"]) != set(ANCHORS)-{group}
            or not r["test_group_was_never_used_for_fit"]
            or r["train_valid_frames"] < 4*MIN_PER_GROUP):
            raise ValueError("Heldout group contaminated training data")
        exp = {(s, "A3_heldout") for s in group_to_members[group]} | {
            (group, "A4_cross_action")}
        act = {(x["subject"],x["evaluation"]) for x in
               r["private_test_participant_rows"]}
        if act != exp or len(exp) != len(r["private_test_participant_rows"]):
            raise ValueError("Missing or duplicate holdout participants")
        winner = min(PARITIES, key=lambda p: (
            r["train_fit_pct_femur"][p], p != "proper"
        ))
        if winner != r["parity_chosen_from_train_only"]:
            raise ValueError("Parity was not selected exclusively on train data")
    cells = []
    for cam in CAMERAS:
        for mod in VARIANTS:
            folds = [r for r in rows
                     if r["camera"] == cam and r["model"] == mod]
            for outcome in OUTCOMES:
                scores = defaultdict(list)
                counts = 0
                test_frames = 0
                for fold in folds:
                    group = fold["heldout_group"]
                    r = [x for x in fold["private_test_participant_rows"]
                         if x["evaluation"] == outcome]
                    expected_count = (len(group_to_members[group])
                                      if outcome == "A3_heldout" else 1)
                    if len(r) != expected_count:
                        raise ValueError("Incorrect participant count for test set")
                    for x in r:
                        counts += 1
                        test_frames += x["valid_test_frames"]
                    for parity in PARITIES:
                        scores[parity].append(mean(x["scores"][parity] for x in r))
                    scores["train_selected"].append(mean(
                        x["selected_training_only"] for x in r))
                proper, reflected = scores["proper"], scores["reflected"]
                select = scores["train_selected"]
                record = {
                    "camera": cam, "model": mod, "evaluation": outcome,
                    "independent_heldout_calibration_groups": 5,
                    "heldout_participants": counts,
                    "valid_six_joint_test_frames": test_frames,
                    "median_heldout_group_mean_proper_pct_femur":
                        round(median(proper),4),
                    "median_heldout_group_mean_reflected_pct_femur":
                        round(median(reflected),4),
                    "median_heldout_group_mean_training_selected_pct_femur":
                        round(median(select),4),
                    "group_count_reflected_better":
                        sum(x > y for x,y in zip(proper,reflected)),
                    "group_count_proper_better":
                        sum(x < y for x,y in zip(proper,reflected)),
                    "training_selected_parity_count": {
                        p: sum(f["parity_chosen_from_train_only"]==p
                               for f in folds)
                        for p in PARITIES
                    },
                }
                if within_person_reference is not None:
                    previous = [
                        within_person_reference[(g,cam,mod)]
                        for g in ANCHORS]
                    for p in PARITIES:
                        record["personal_A3_fitted_anchor_median_"+p+"_pct_femur"] = (
                            round(median(
                                x["fits"][p][outcome][
                                    "mean_discrepancy_pct_reference_femur_length"]
                                for x in previous
                            ),4))
                cells.append(record)
    return {
        "status": "complete_exploratory_LOCO_A3_A4_3d_shape_transfer",
        "train": "4 OTHER calibration group anchors, A3 first 12 frames each",
        "test_A3": "all participants in heldout calibration group, last 12 A3 frames",
        "test_A4": "heldout group anchor, all 24 A4 frames",
        "no_holdout_group_subject_or_session_in_fit": True,
        "calibration_groups": 5,
        "new_independent_calibration_groups": 0,
        "A3_subjects_evaluated": 26,
        "A4_anchor_subjects_evaluated": 5,
        "model_view_LOCO_folds": 20,
        "heldout_model_participant_action_evaluations": 124,
        "cells": cells,
        "not_absolute_MPJPE_mm_or_clinical_accuracy": True,
        "GT_hip_root_uses_Vicon_test_landmarks": True,
        "model_training_data_overlap_not_proven_absent": True,
        "person_generalization_test_within_existing_public_dataset_only": True,
        "source_raw_images_and_landmarks_published": False,
    }


def run_study(root, model_dir, analysis_dir, private_output, aggregate_output):
    """Read originals in place; never save model/Vicon per-frame coordinates."""
    repo = Path(__file__).resolve().parents[2]
    dest = [private_output.resolve(), aggregate_output.resolve()]
    source = analysis_dir.resolve()
    if (dest[0] == dest[1] or
        any(p.exists() or p.parent != source or
            p == repo or repo in p.parents for p in dest)):
        raise ValueError("Existing output or output outside authorized private analysis directory")
    census = json.loads((source / "A3_26subject_D2_census_protocol_20261116.json").read_text("utf-8-sig"))
    map_ = json.loads((root / "calibration/external/subject_map.json").read_text("utf-8-sig"))
    groups = validate_groups(map_, census)
    prior_folder = source / "mediapipe_parity_A3_A4_20261117"
    prior = {}
    for sub in ANCHORS:
        for cam in CAMERAS:
            for model in VARIANTS:
                j = json.loads((prior_folder / f"parity_{sub}_{cam}_{model}.json").read_text("utf-8-sig"))
                if (j["subject"],j["camera"],j["model"])!=(sub,cam,model):
                    raise ValueError("Prior per-person A3 comparator identity mismatch")
                prior[(sub,cam,model)] = j
    models = {}
    for model in VARIANTS:
        path = model_dir / f"pose_landmarker_{model}.task"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != census["model_variants"][model]["sha256"]:
            raise ValueError("Frozen model SHA does not match A3 census protocol")
        models[model] = path

    rows=[]
    images=0
    detections=0
    for cam in CAMERAS:
        for model in VARIANTS:
            frame_map={}
            for sub in TOTAL_SUBJECTS:
                frames,quality = read_pose_frames(
                    root, models[model], "A3", cam, sub)
                if (quality["model_images_attempted"] != SAMPLE_COUNT or
                    quality["selected_jpg_names_different_from_previous"]):
                    raise ValueError("A3 source RGB protocol changed")
                frame_map[sub]={"A3":(frames,quality)}
                images+=quality["model_images_attempted"]
                detections+=quality["pose_detections"]
                if sub in ANCHORS:
                    deadlift,qa = read_pose_frames(
                        root, models[model], "A4", cam, sub)
                    if (qa["model_images_attempted"] != SAMPLE_COUNT or
                        qa["selected_jpg_names_different_from_previous"]):
                        raise ValueError("A4 source RGB protocol changed")
                    old = prior[(sub,cam,model)]
                    for action,q in (("A3",quality),("A4",qa)):
                        expected=old[f"{action}_source_audit"]
                        for key in ("model_images_attempted","pose_detections",
                                    "valid_six_joint_frames"):
                            if q[key]!=expected[key]:
                                raise ValueError("Changed frozen MediaPipe inference reference")
                    frame_map[sub]["A4"]=(deadlift,qa)
                    images+=qa["model_images_attempted"]
                    detections+=qa["pose_detections"]
            for group in ANCHORS:
                outcome = evaluate_excluded_group(frame_map, groups[group], group)
                outcome["camera"]=cam
                outcome["model"]=model
                rows.append(outcome)
            print(json.dumps({
                "progress": "completed_camera_model_cell",
                "camera": cam, "model": model,
                "completed_folds": len(rows),
                "expected_folds": 20,
            }),flush=True)
    report = summarize(rows,groups,prior)
    report["model_image_inferences"]=images
    report["pose_detections"]=detections
    report["frozen_model_SHA256_verified"]=True
    if images != (26+5)*len(CAMERAS)*len(VARIANTS)*SAMPLE_COUNT:
        raise ValueError("Wrong total image-model inference count")
    if detections <= 0 or detections > images:
        raise ValueError("Invalid detection count")
    private = {
        "status":report["status"],
        "rows":rows,
        "calibration_group_sizes":{g:len(groups[g]) for g in ANCHORS},
        "model_asset_sha256": {
            k:hashlib.sha256(v.read_bytes()).hexdigest()
            for k,v in models.items()
        },
        "private_local_only":True,
        "no_per_frame_coordinates_or_RGB_saved":True,
        "not_absolute_MJPPE":True
    }
    for dest,payload in ((private_output,private),(aggregate_output,report)):
        with dest.open("x",encoding="utf-8") as f:
            json.dump(payload,f,ensure_ascii=False,indent=2)
            f.write("\\n")
    print(json.dumps({k:report[k] for k in (
        "status","A3_subjects_evaluated","A4_anchor_subjects_evaluated",
        "heldout_model_participant_action_evaluations","model_image_inferences",
        "pose_detections","calibration_groups",
    )}),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--private-output",type=Path,required=True)
    parser.add_argument("--aggregate-output",type=Path,required=True)
    args=parser.parse_args(argv)
    run_study(args.root,args.model_dir,args.analysis_dir,
              args.private_output,args.aggregate_output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
