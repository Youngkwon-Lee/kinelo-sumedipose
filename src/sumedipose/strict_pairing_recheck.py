"""Source-faithful A3 D2 census / A4 D2 anchor angular-error sensitivity.

Images and model inference inputs are held fixed. Original and corrected Vicon
coordinate references are evaluated against each SAME inferred pose, in memory.
Participant-level outputs remain in approved local research storage, outside Git.
Descriptive calibration correlations use 5 shared calibration groups, not frames.
No clinical validity, independent calibration, or physical exposure sync claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median, mean
from zipfile import ZipFile

import numpy as np

from .benchmark import JOINT_NAMES, _angles_from_named, summarize_errors, unique_sample_indices
from .calibration_strata import (
    SESSIONS, CAMERAS, VARIANTS, MODALITIES,
    _get_calibration, spearman_descriptive, weighted_two_side_mae,
)
from .geometry import joint_angle
from .io import read_sample
from .position_holdout import strict_unique_pairs

FRAMES = 24
ACTIONS = ("A3", "A4")
COHORT_LABELS = ("anchor5_D2", "disjoint10_D2", "remaining11_D2", "all26_D2")


def dataset_cohorts(census: dict, new_subjects: dict, subject_group_map: dict) -> dict:
    """Private-only participant mapping, never returned in aggregate reports."""
    excluded = set(census["missing_subjects"])
    all26 = {f"S{i}" for i in range(1, 29)} - excluded
    anchor = set(SESSIONS)
    disjoint = {s for g in new_subjects["subject_groups"] for s in g["subjects"]}
    remainder = all26 - anchor - disjoint
    groups = {s: subject_group_map.get(s) for s in all26}
    if (len(all26) != 26 or len(anchor) != 5 or len(disjoint) != 10
            or len(remainder) != 11 or anchor & disjoint
            or any(g not in SESSIONS for g in groups.values())):
        raise ValueError("26-person frozen cohorts or five calibration groups differ")
    sizes = {g: sum(groups[s] == g for s in all26) for g in SESSIONS}
    if sizes != census["calibration_group_sizes"]:
        raise ValueError("Original five-group participant composition changed")
    cohorts = {"anchor5_D2": anchor, "disjoint10_D2": disjoint,
               "remaining11_D2": remainder, "all26_D2": all26}
    for label, members in cohorts.items():
        if set(groups[s] for s in members) != set(SESSIONS):
            raise ValueError(f"Missing calibration group in cohort {label}")
    return {"subjects": all26, "cohorts": cohorts, "group_map": groups}


def source_baseline(analysis_dir: Path, action: str, sub: str, cam: str, model: str) -> dict:
    if action == "A4":
        candidates = [analysis_dir / f"A4D2_pose_{sub}_{cam}_{model}.json"]
    else:
        candidates = [
            analysis_dir / f"S1A3_pose_{sub}_{cam}_{model}.json",
            analysis_dir / f"A3_generalization_{sub}_D2_{cam}_{model}.json",
            analysis_dir / f"A3_censusD2_{sub}_{cam}_{model}.json",
        ]
    present = [p for p in candidates if p.is_file()]
    if len(present) != 1:
        raise ValueError("Missing/ambiguous original local aggregate report")
    return json.loads(present[0].read_text(encoding="utf-8-sig"))


def paired_evaluate(root, model_asset_path, action, subject, camera,
                    *, max_frames=FRAMES, min_visibility=.5):
    """Single real MediaPipe prediction supports both old/new GT error metrics."""
    if action not in ACTIONS or camera not in CAMERAS or max_frames != FRAMES:
        raise ValueError("Non-frozen action/view/image count")
    import cv2
    import mediapipe as mp

    paths, world, cam = read_sample(
        root, action=action, subject=subject, camera=camera, repeat="D2")
    if len(world) != len(cam):
        raise ValueError("World / camera record-index lengths differ")
    with ZipFile(paths["rgb"]) as archive:
        old, unavailable = unique_sample_indices(cam, archive.namelist(), max_frames)
        strict, qa = strict_unique_pairs(cam, archive.namelist(), max_frames)
        if (len(old) != max_frames or
                [name for name, _ in old] != [name for name, _ in strict]):
            raise ValueError("Images differ; cannot use fixed-frame paired sensitivity")
        if unavailable != qa["missing_original_jpg_groups"]:
            raise ValueError("Unexpected image-source missingness")
        for (old_name, old_idx), (new_name, new_idx) in zip(old, strict):
            if old_name != new_name:
                raise ValueError("Mismatched original image selection")
            if cam[new_idx].get("factor") != cam[new_idx].get("dup_num"):
                raise ValueError("Invalid strict coordinate association")
        errors = {
            version: {
                dimension: {side: [] for side in JOINT_NAMES}
                for dimension in MODALITIES
            }
            for version in ("original", "strict")
        }
        detections = 0
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_asset_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as task:
            for (name, original_index), (_, strict_index) in zip(old, strict):
                jpg = cv2.imdecode(
                    np.frombuffer(archive.read(name), np.uint8),
                    cv2.IMREAD_COLOR)
                if jpg is None:
                    continue
                h, w = jpg.shape[:2]
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB))
                result = task.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.pose_landmarks or not result.pose_world_landmarks:
                    continue
                detections += 1
                xy = result.pose_landmarks[0]
                xyz = result.pose_world_landmarks[0]
                for side, part in JOINT_NAMES.items():
                    ids = part["pred"]
                    if any(xy[i].visibility < min_visibility or
                           xyz[i].visibility < min_visibility for i in ids):
                        continue
                    pred2 = [[xy[i].x * w, xy[i].y * h] for i in ids]
                    pred3 = [[xyz[i].x, xyz[i].y, xyz[i].z] for i in ids]
                    p2, p3 = joint_angle(*pred2), joint_angle(*pred3)
                    for version, record_index in (
                        ("original", original_index), ("strict", strict_index)
                    ):
                        try:
                            gt2 = _angles_from_named(
                                cam[record_index], part["reference"], "xy")
                            gt3 = _angles_from_named(
                                world[record_index], part["reference"], "xyz")
                        except (ValueError, IndexError, KeyError, TypeError):
                            continue
                        if np.isfinite(gt2) and np.isfinite(p2):
                            errors[version]["image_2d"][side].append(abs(p2 - gt2))
                        if np.isfinite(gt3) and np.isfinite(p3):
                            errors[version]["world_3d"][side].append(abs(p3 - gt3))
    return {
        "sample": f"{camera}{subject}{action}D2",
        "model_variant": Path(model_asset_path).stem.replace("pose_landmarker_", ""),
        "model_asset_sha256": hashlib.sha256(Path(model_asset_path).read_bytes()).hexdigest(),
        "mediapipe_version": mp.__version__,
        "inference_mode": "IMAGE",
        "max_unique_images_requested": max_frames,
        "unique_images_sampled": max_frames,
        "model_detected_poses": detections,
        "pairing": {
            "original_wrong_coordinate_references": qa["previous_index_selection_invalid_pairs"],
            "strict_source_missing_jpg_groups": qa["missing_original_jpg_groups"],
            "ambiguous_reference_groups": qa["ambiguous_or_unverifiable_jpg_groups"],
            "changed_jpg_name_count": qa["selected_jpg_names_different_from_previous"],
            "frozen_same_rgb_frames": True,
            "new_reference_uses_dup_num_equals_factor": True,
        },
        "original_results": {
            dim: {side: summarize_errors(errors["original"][dim][side], max_frames)
                  for side in JOINT_NAMES}
            for dim in MODALITIES
        },
        "strict_results": {
            dim: {side: summarize_errors(errors["strict"][dim][side], max_frames)
                  for side in JOINT_NAMES}
            for dim in MODALITIES
        },
        "physical_exposure_time_synchronization_verified": False,
        "clinical_joint_angle_accuracy_validated": False,
        "individual_rgb_or_landmark_coordinates_saved": False,
    }


def compare_original_replay(replay: dict, previous: dict, *, tolerance=.005):
    """Fail closed if the old reference cannot be recreated on identical pixels."""
    if (replay["sample"] != previous["sample"] or
            replay["model_variant"] != previous["model_variant"] or
            replay["model_asset_sha256"] != previous["model_asset_sha256"] or
            replay["model_detected_poses"] != previous["model_detected_poses"]):
        raise ValueError("Original run identity or detection counts differ")
    for space in MODALITIES:
        for side in JOINT_NAMES:
            actual = replay["original_results"][space][side]
            earlier = previous["results"][space][side]
            if actual["valid"] != earlier["valid"] or actual["attempted"] != earlier["attempted"]:
                raise ValueError("Original model output visibility changed")
            x, y = actual["MAE_degrees"], earlier["MAE_degrees"]
            if (x is None) != (y is None) or (
                x is not None and abs(x-y) > tolerance
            ):
                raise ValueError("Original angular result not reproduced")
    return True


def _report_metrics(report, version, modality):
    return weighted_two_side_mae(
        {"results": report[f"{version}_results"]},
        modality, expected_frames=FRAMES)


def _group_correlation(members, group_map, reports, calibration, *,
                       action, cam, model, modality, version):
    by_group = defaultdict(list)
    individual_mae = []
    valid = 0
    for subject in sorted(members, key=lambda s: int(s[1:])):
        item = reports[(action, subject, cam, model)]
        score, count = _report_metrics(item, version, modality)
        if not count or score is None:
            raise ValueError("Missing valid angles for a participant")
        valid += count
        by_group[group_map[subject]].append(score)
        individual_mae.append(score)
    if set(by_group) != set(SESSIONS):
        raise ValueError("Missing independent calibration group")
    x = [calibration[g] for g in SESSIONS]
    ys = [mean(by_group[g]) for g in SESSIONS]
    rho = spearman_descriptive(x, ys)
    if rho is None:
        raise ValueError("Undefined calibration group association")
    loo = [spearman_descriptive(
        [x[j] for j in range(5) if j != i],
        [ys[j] for j in range(5) if j != i]
    ) for i in range(5)]
    return {
        "rho": rho,
        "loo_rho_range": [min(loo), max(loo)] if all(v is not None for v in loo) else None,
        "participant_median_mae_degrees": round(median(individual_mae), 4),
        "session_equal_weight_median_mae_degrees": round(median(ys), 4),
        "n_valid_side_joint_measurements": valid,
    }


def aggregate_strict(protocol, prior_cohort, groupmap, new_cohort, calibration,
                     reports, *, original_analysis_name=None):
    """Only cross-group summaries: original-vs-strict contrasts, 40 outcomes."""
    cohorts = dataset_cohorts(protocol, new_cohort, groupmap)
    groups = cohorts["group_map"]
    cal = _get_calibration(calibration)
    expected = {
        ("A3", s, cam, mod)
        for s in cohorts["subjects"] for cam in CAMERAS for mod in VARIANTS
    } | {
        ("A4", s, cam, mod)
        for s in SESSIONS for cam in CAMERAS for mod in VARIANTS
    }
    if set(reports) != expected or len(reports) != 124:
        raise ValueError("Require 104 A3 + 20 A4 validated reports")
    if any(r["pairing"]["changed_jpg_name_count"] or
           not r["pairing"]["frozen_same_rgb_frames"] or
           r["unique_images_sampled"] != FRAMES
           for r in reports.values()):
        raise ValueError("Original image selection must remain frozen")
    results = []
    for action, subsets in (
        ("A3", cohorts["cohorts"]),
        ("A4", {"anchor5_D2": set(SESSIONS)}),
    ):
        for cohort_label, members in subsets.items():
            for cam in CAMERAS:
                for model in VARIANTS:
                    for modality in MODALITIES:
                        old = _group_correlation(
                            members, groups, reports, cal,
                            action=action, cam=cam, model=model,
                            modality=modality, version="original")
                        fixed = _group_correlation(
                            members, groups, reports, cal,
                            action=action, cam=cam, model=model,
                            modality=modality, version="strict")
                        if action == "A3":
                            match = [
                                v["cohorts"][cohort_label]["rho_descriptive"]
                                for v in prior_cohort["per_outcome"]
                                if (v["camera"],v["model"],v["reference_space"])
                                == (cam, model, modality)
                            ]
                            if len(match) != 1 or match[0] != old["rho"]:
                                raise ValueError("Old cohort association not reproduced")
                        results.append({
                            "action": action,
                            "cohort": cohort_label,
                            "camera": cam,
                            "model": model,
                            "reference_space": modality,
                            "n_participants": len(members),
                            "independent_calibration_groups": 5,
                            "original_rho": old["rho"],
                            "source_valid_rho": fixed["rho"],
                            "delta_rho": round(fixed["rho"] - old["rho"], 5),
                            "original_participant_median_MAE_deg":
                                old["participant_median_mae_degrees"],
                            "source_valid_participant_median_MAE_deg":
                                fixed["participant_median_mae_degrees"],
                            "delta_participant_median_MAE_deg": round(
                                fixed["participant_median_mae_degrees"] -
                                old["participant_median_mae_degrees"], 4),
                            "source_valid_leave_one_group_out_rho_range":
                                fixed["loo_rho_range"],
                            "valid_side_observations_source_valid":
                                fixed["n_valid_side_joint_measurements"],
                        })
    invalid_by_action = {
        act: sum(reports[(act, sub, cam, "lite")]["pairing"][
            "original_wrong_coordinate_references"]
            for sub in (cohorts["subjects"] if act=="A3" else SESSIONS)
            for cam in CAMERAS)
        for act in ACTIONS
    }
    output = {
        "status": "complete_exploratory_strict_original_frame_pairing_sensitivity",
        "study": "A3 26 available D2 participants and A4 five D2 anchors: source GT index correction",
        "frozen_images_per_recording": FRAMES,
        "n_A3_participants": 26,
        "n_A4_anchor_participants": 5,
        "model_reports_checked": len(reports),
        "model_image_inferences": len(reports) * FRAMES,
        "pose_detections": sum(r["model_detected_poses"] for r in reports.values()),
        "distinct_participant_camera_action_rgb_samples": 62,
        "original_wrong_unique_jpg_coordinate_pairings": invalid_by_action,
        "total_old_wrong_pairings": sum(invalid_by_action.values()),
        "images_changed_after_strict_matching": 0,
        "independent_calibration_groups": 5,
        "new_independent_calibration_groups": 0,
        "n_group_descriptive_associations": len(results),
        "comparison": results,
        "all_original_model_aggregates_reproduced": True,
        "prior_26_A3_cohort_rhos_reproduced": True,
        "model_training_overlap_excluded": False,
        "physical_exposure_timestamps_verified": False,
        "reference_anatomical_joint_centre_equivalence_verified": False,
        "absolute_3d_MPJPE_mm_validated": False,
        "clinical_validity_established": False,
        "all_model_images_and_per_person_results_private": True,
        "interpretation": [
            "Original source JPGs and model variants are identical; only Vicon index mapping differs",
            "The same five calibration groups are reused; 26 participants are not 26 independent calibrations",
            "n=5 rho can jump discretely with small reference index changes and changing participant composition",
            "G65 stereo calibration residual is not proven cause of monocular 3D angular error",
            "The 2D projected angle and 3D WCS included angle have distinct reference geometries",
            "No physical camera-to-Vicon timing or anatomical joint centre equivalence verified",
            "This does not report globally registered 3D joint XYZ or MPJPE in millimetres",
        ],
        "raw_images_or_per_participant_source_data_published": False,
    }
    if len(results) != 40:
        raise ValueError("Expected 4 A3 cohorts and one A4 cohort times eight outcomes")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-reports", type=int, default=124,
                        help="Bounded preflight then resume; defaults to full 124")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    folder, output = args.analysis_dir.resolve(), args.output_dir.resolve()
    if (output == repo or repo in output.parents or
            output != folder / "strict_source_pairing_20261117" or
            not 1 <= args.max_new_reports <= 124):
        parser.error("Private output must be a fixed local analysis subfolder")
    census = json.loads((folder / "A3_26subject_D2_census_protocol_20261116.json").read_text(encoding="utf-8-sig"))
    prior = json.loads((folder / "A3_26subject_D2_cohort_composition_sensitivity_20261116.json").read_text(encoding="utf-8-sig"))
    source_groups = json.loads((args.root / "calibration/external/subject_map.json").read_text(encoding="utf-8-sig"))
    new_cohort = json.loads((folder / "A3_new_subject_generalization_protocol_20261116.json").read_text(encoding="utf-8-sig"))
    calibration = json.loads((folder / "S1_S6_S11_S17_S23_calibration_five_session_overview.json").read_text(encoding="utf-8-sig"))
    cohorts = dataset_cohorts(census, new_cohort, source_groups)
    models = {}
    for model in VARIANTS:
        f = args.model_dir / f"pose_landmarker_{model}.task"
        if hashlib.sha256(f.read_bytes()).hexdigest() != census["model_variants"][model]["sha256"]:
            parser.error("Original model weight SHA differs")
        models[model] = f
    output.mkdir(parents=True, exist_ok=True)
    all_keys = [(action, subject, cam, model)
                for action, subjects in (("A3", cohorts["subjects"]),("A4", set(SESSIONS)))
                for subject in sorted(subjects, key=lambda s:int(s[1:]))
                for cam in CAMERAS for model in VARIANTS]
    if len(all_keys) != 124:
        raise ValueError("Expected 124 model-view-subject recordings")
    reports = {}
    new_count = 0
    for key in all_keys:
        act, sub, cam, model = key
        out = output / f"strict_{act}D2_{sub}_{cam}_{model}.json"
        original = source_baseline(folder, act, sub, cam, model)
        if out.exists():
            report = json.loads(out.read_text(encoding="utf-8-sig"))
        elif new_count < args.max_new_reports:
            report = paired_evaluate(args.root, models[model], act, sub, cam)
            compare_original_replay(report, original)
            with out.open("x", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            new_count += 1
        else:
            continue
        compare_original_replay(report, original)
        if (report["model_asset_sha256"] != census["model_variants"][model]["sha256"]
                or report["pairing"]["changed_jpg_name_count"] != 0
                or report["pairing"]["ambiguous_reference_groups"] != 0):
            raise ValueError("Report source or model identity changed")
        reports[key] = report
    if len(reports) != len(all_keys):
        print(json.dumps({"status":"incomplete_source_valid_recheck",
            "private_model_reports_current":len(reports),
            "expected_model_reports":len(all_keys),
            "new_reports_this_invocation":new_count}))
        return 0
    result = aggregate_strict(census, prior, source_groups, new_cohort, calibration, reports)
    dest = output / "strict_A3_26_A4_5_D2_aggregate_20261117.json"
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8-sig")) != result:
            raise ValueError("Existing aggregate differs; preserved")
    else:
        with dest.open("x", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(json.dumps({k:result[k] for k in (
        "status","model_reports_checked","model_image_inferences",
        "pose_detections","original_wrong_unique_jpg_coordinate_pairings",
        "n_group_descriptive_associations")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
