"""Local-only exploratory association between stereo calibration and pose errors.

Unit of statistical analysis is the calibration session/subject, NOT video frames.
The calibration residual is from stereo pair C6-C5 while the pose models are
monocular. Their association is confounded by camera view and subject. Neither
causal inference nor a clinically accepted quality cutoff is supported.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

SESSIONS = ("S1", "S6", "S11", "S17", "S23")
CAMERAS = ("C1", "C6")
VARIANTS = ("lite", "full")
MODALITIES = ("image_2d", "world_3d")


def _finite_nonnegative(value, label):
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"Invalid nonnegative {label}: {value}")
    return x


def _ranks(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    result = [0.0] * len(indexed)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        rank = (pos + 1 + end) / 2
        for idx, _ in indexed[pos:end]:
            result[idx] = rank
        pos = end
    return result


def spearman_descriptive(x, y):
    """Session-level Spearman rho, no hypothesis-test p-value."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    a, b = sum(rx) / len(rx), sum(ry) / len(ry)
    numer = sum((v - a) * (w - b) for v, w in zip(rx, ry))
    denom = math.sqrt(sum((v - a) ** 2 for v in rx) * sum((w - b) ** 2 for w in ry))
    return round(numer / denom, 5) if denom else None


def weighted_two_side_mae(record, modality, expected_frames=24):
    """Within one session, compute valid-side weighted 2D *or* 3D MAE."""
    data = record["results"][modality]
    total, n = 0.0, 0
    for side in ("left", "right"):
        item = data[side]
        count = int(item["valid"])
        if count < 0 or count > expected_frames:
            raise ValueError("Landmark count exceeds sampled images")
        mae = item["MAE_degrees"]
        if count and mae is None:
            raise ValueError("Valid landmarks but missing angle MAE")
        if count:
            total += _finite_nonnegative(mae, "angular MAE") * count
            n += count
    return (round(total / n, 5), n) if n else (None, 0)


def _get_calibration(calibration):
    groups = {}
    for row in calibration["comparisons"]:
        if row["pair"] != "G65":
            continue
        session = row["session"]
        if session in groups:
            raise ValueError(f"Duplicate stereo calibration session: {session}")
        groups[session] = _finite_nonnegative(row["sampson_median_px"], "G65 Sampson residual")
    missing = set(SESSIONS) - set(groups)
    if missing:
        raise ValueError(f"Missing calibration sessions: {sorted(missing)}")
    return groups


def analyse(calibration, pose_reports, protocol, cutoff_px=5.0):
    """Return aggregate-only descriptive results; never publish per-subject metrics."""
    cutoff = _finite_nonnegative(cutoff_px, "posthoc threshold")
    if protocol["max_unique_frames_per_view"] != 24 or protocol["repeat"] != "D2":
        raise ValueError("This analysis requires existing 24-unique-JPEG A3 D2 protocol")
    calibration_by_session = _get_calibration(calibration)
    expected_hash = {v: protocol["model_variants"][v]["sha256"] for v in VARIANTS}
    # A session must have BOTH models and BOTH views to prevent changing sample
    # composition across rows when one evaluation is missing.
    included, missing_files = [], []
    for session in SESSIONS:
        keys = [(session, cam, variant) for cam in CAMERAS for variant in VARIANTS]
        absent = [f"S1A3_pose_{s}_{cam}_{variant}.json" for s, cam, variant in keys if (s, cam, variant) not in pose_reports]
        if absent:
            missing_files.extend(absent)
        else:
            included.append(session)
    for (session, camera, variant), record in pose_reports.items():
        if (session not in SESSIONS or camera not in CAMERAS or variant not in VARIANTS):
            raise ValueError("Unexpected pose result key")
        if record["sample"] != f"{camera}{session}A3D2" or record["model_variant"] != variant:
            raise ValueError("Mismatched sample or model variant")
        if record["model_asset_sha256"] != expected_hash[variant]:
            raise ValueError(f"Model asset SHA mismatch: {variant}")
        if (record["unique_images_sampled"] != 24 or record["max_unique_images_requested"] != 24
                or record["inference_mode"] != "IMAGE"):
            raise ValueError("Incompatible sampling or inference protocol")
    associations = []
    for cam in CAMERAS:
        for variant in VARIANTS:
            for modality in MODALITIES:
                session_maes = []
                for ses in included:
                    mae, count = weighted_two_side_mae(pose_reports[(ses,cam,variant)], modality)
                    if mae is not None:
                        session_maes.append((ses, mae))
                xs = [calibration_by_session[s] for s, _ in session_maes]
                ys = [v for _, v in session_maes]
                high = [v for s, v in session_maes if calibration_by_session[s] > cutoff]
                low = [v for s, v in session_maes if calibration_by_session[s] <= cutoff]
                associations.append({
                    "camera":cam, "model":variant, "reference_space":modality,
                    "session_count":len(session_maes),
                    "spearman_rho_descriptive":spearman_descriptive(xs, ys),
                    "session_equal_weight_mae_median_deg":round(median(ys),4) if ys else None,
                    "posthoc_gt_cutoff_count":len(high),
                    "posthoc_le_cutoff_count":len(low),
                    "posthoc_gt_cutoff_mae_median_deg":round(median(high),4) if len(high)>=2 else None,
                    "posthoc_le_cutoff_mae_median_deg":round(median(low),4) if len(low)>=2 else None,
                    "group_comparison_validated":False,
                })
    contrasts = []
    for variant in VARIANTS:
        for modality in MODALITIES:
            pairs=[]
            for ses in included:
                a, _ = weighted_two_side_mae(pose_reports[(ses,"C1",variant)], modality)
                b, _ = weighted_two_side_mae(pose_reports[(ses,"C6",variant)], modality)
                if a is not None and b is not None:
                    pairs.append((ses,b-a))
            contrasts.append({
                "model":variant,"reference_space":modality,
                "session_count":len(pairs),
                "c6_minus_c1_mae_median_deg":round(median(delta for _,delta in pairs),4) if pairs else None,
                "g65_vs_c6_minus_c1_spearman_rho_descriptive":spearman_descriptive(
                    [calibration_by_session[ses] for ses,_ in pairs],
                    [delta for _,delta in pairs]
                ),
                "view_difference_is_not_a_causal_adjustment":True,
            })
    valid_all = len(included)==len(SESSIONS) and all(
        row["session_count"]==len(SESSIONS) for row in associations+contrasts
    )
    return {
        "status":"complete_exploratory" if valid_all else "incomplete_missing_or_invalid_sessions",
        "calibration_pair":"G65 (C6-C5)",
        "expected_session_count":len(SESSIONS), "complete_session_count":len(included),
        "missing_pose_report_files":missing_files,
        "sessions_used_for_roi_tuning":["S1"],
        "sessions_outside_roi_tuning":[s for s in included if s != "S1"],
        "posthoc_high_residual_cutoff_px":cutoff,
        "cutoff_was_selected_after_observing_calibration_data":True,
        "session_equal_weights_not_frame_pooled":True,
        "associations":associations,
        "camera_view_contrasts":contrasts,
        "inference_limitations":[
            "n<=5 session/subject codes; no statistical significance or causal inference",
            "Camera-view C6 minus C1 is descriptive and does not adjust for anatomical projection or session confounders",
            "Calibration G65 is a stereo-camera residual, not a direct measure of monocular pose accuracy",
            "2D projected-Vicon and 3D WCS angular errors must not be ranked against each other",
            "MediaPipe hip/knee/ankle landmarks differ from Vicon anatomical joint centres",
            "S1 was used for prior ROI tuning; not independent heldout evidence",
            "Predictions are not audited for training-data overlap or clinical validity",
        ],
        "clinical_accuracy_validated":False,
        "physical_camera_vicon_timestamp_sync_verified":False,
        "raw_images_or_subject_level_pose_metrics_exported":False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cutoff-px",type=float,default=5.0)
    args=parser.parse_args(argv)
    source=args.analysis_dir
    calibration=json.loads((source/"S1_S6_S11_S17_S23_calibration_five_session_overview.json").read_text(encoding="utf8"))
    protocol=json.loads((source/"S1A3_mediapipe_lite_full_protocol.json").read_text(encoding="utf8"))
    pose={}
    for s in SESSIONS:
        for c in CAMERAS:
            for v in VARIANTS:
                f=source/f"S1A3_pose_{s}_{c}_{v}.json"
                if f.is_file():pose[(s,c,v)]=json.loads(f.read_text(encoding="utf8"))
    report=analyse(calibration,pose,protocol,args.cutoff_px)
    # The destination MUST remain outside the public code checkout.
    repo=Path(__file__).resolve().parents[2]
    dest=args.output.resolve()
    if dest==repo or repo in dest.parents:
        parser.error("Private derived reports cannot be written inside the public source checkout")
    if dest.exists():parser.error("Output exists; refusing to overwrite")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({k:report[k] for k in ("status","complete_session_count","expected_session_count","missing_pose_report_files")},ensure_ascii=False))
    return 0 if report["status"]=="complete_exploratory" else 2


if __name__=="__main__":
    raise SystemExit(main())
