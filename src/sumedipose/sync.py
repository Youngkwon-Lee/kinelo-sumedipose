"""Local-only structural audit of original SUMediPose A3 WCS/camera/RGB records.

The camera archive supplies image-path associations. This is not a validation of
physical capture timestamps or camera/Vicon hardware clock synchronization.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from .geometry import knee_angles
from .io import read_sample

_IMAGE_FRAME = re.compile(r"_([0-9]+)\.jpg$", re.IGNORECASE)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return None
    return round(float(np.corrcoef(a[good], b[good])[0, 1]), 9)


def _angles(records: list, side: str) -> np.ndarray:
    vals = []
    for record in records:
        try:
            vals.append(float(knee_angles(record)[f"{side}_knee_degrees"]))
        except (KeyError, IndexError, ValueError, TypeError):
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def _angle_metrics(wcs: list, camera: list) -> dict:
    by_side = {}
    for side in ("left", "right"):
        a, b = _angles(wcs, side), _angles(camera, side)
        finite = np.isfinite(a) & np.isfinite(b)
        entry = {"comparable_records": int(finite.sum()), "mean_abs_diff_degrees": None,
                 "max_abs_diff_degrees": None, "same_index_correlation": None,
                 "previous_index_correlation": None, "next_index_correlation": None}
        if finite.any():
            diffs = np.abs(a[finite] - b[finite])
            entry["mean_abs_diff_degrees"] = round(float(np.mean(diffs)), 6)
            entry["max_abs_diff_degrees"] = round(float(np.max(diffs)), 6)
            entry["same_index_correlation"] = _safe_corr(a, b)
            if len(a) > 1:
                entry["previous_index_correlation"] = _safe_corr(a[1:], b[:-1])
                entry["next_index_correlation"] = _safe_corr(a[:-1], b[1:])
        by_side[side] = entry
    return by_side


def audit_sample(root: str | Path, camera: str = "C1",
                 subject: str = "S1", repeat: str = "D1") -> tuple[dict, list[dict]]:
    """Return aggregate findings and local-only per-record map; never extract raw ZIPs.

    WCS and camera XYZ comparisons only test index consistency of 3D records.
    Their agreement does NOT independently establish RGB exposure timestamps.
    """
    paths, world, cam = read_sample(root, camera=camera, subject=subject, action="A3", repeat=repeat)
    with ZipFile(paths["rgb"]) as archive:
        image_names = {i.filename for i in archive.infolist() if not i.is_dir()}
    zip_prefix = f"frames/{camera}/{subject}/{camera}{subject}A3{repeat}/"
    label = f"{camera}{subject}A3{repeat}"

    w_ids = [r.get("frame_num") for r in world]
    cam_ids = [r.get("frame_num") for r in cam]
    wd_ids = [r.get("frame_num_w_drop") for r in cam]
    marker_ids_match = len(world) == len(cam) and all(
        w.get("point_ids") == c.get("point_ids") for w, c in zip(world, cam)
    )
    indexed = len(world) == len(cam) and all(
        type(frame) is int and frame == index for index, frame in enumerate(w_ids)
    )

    rows = []
    referenced = []
    path_mismatches = 0
    for index, record in enumerate(cam):
        rel = record.get("path", "")
        basename = Path(rel).name
        expected = f"{zip_prefix}{basename}"
        match = _IMAGE_FRAME.search(basename)
        fname_ok = bool(
            isinstance(rel, str) and rel.replace("\\", "/") == expected
            and match and type(record.get("frame_num")) is int
            and int(match.group(1)) == record["frame_num"]
            and basename.startswith(label + "_")
        )
        if not fname_ok:
            path_mismatches += 1
        referenced.append(basename)
        rows.append({
            "record_index": index,
            "wcs_frame_num": w_ids[index] if index < len(w_ids) else None,
            "camera_frame_num_w_drop": record.get("frame_num_w_drop"),
            "camera_frame_num": record.get("frame_num"),
            "rgb_member": basename,
            "rgb_exists_in_archive": basename in image_names,
            "rgb_reused": False,
        })

    counts = Counter(referenced)
    for row in rows:
        row["rgb_reused"] = counts[row["rgb_member"]] > 1

    image_missing = set(referenced) - image_names
    image_unreferenced = image_names - set(referenced)
    cam_numeric = all(type(x) is int for x in cam_ids)
    wd_numeric = all(type(x) is int for x in wd_ids)
    camera_nondecreasing = cam_numeric and all(a <= b for a, b in zip(cam_ids, cam_ids[1:]))
    wdrop_consecutive = wd_numeric and all(b - a == 1 for a, b in zip(wd_ids, wd_ids[1:]))
    unique_camera_numbers = sorted(set(cam_ids)) if cam_numeric else []
    camera_unique_contiguous = bool(unique_camera_numbers) and all(
        b - a == 1 for a, b in zip(unique_camera_numbers, unique_camera_numbers[1:])
    )

    angles = _angle_metrics(world, cam) if world and cam else {}
    angle_ok = bool(angles) and all(
        m["comparable_records"] == len(world)
        and m["mean_abs_diff_degrees"] is not None
        and m["mean_abs_diff_degrees"] < 1.0
        for m in angles.values()
    )
    # Coordinate record consistency is distinct from RGB member availability.
    # Missing terminal RGBs must not invalidate matching WCS/camera 3D indices.
    record_index_ok = bool(indexed and marker_ids_match and angle_ok)
    rgb_mapping_ok = bool(path_mismatches == 0 and not image_missing and not image_unreferenced)
    rgb_missing_rows = [r["record_index"] for r in rows if not r["rgb_exists_in_archive"]]
    complete_audit_ok = bool(
        record_index_ok and rgb_mapping_ok and camera_nondecreasing
        and wdrop_consecutive and camera_unique_contiguous
    )
    report = {
        "sample": label,
        "wcs_records": len(world),
        "camera_records": len(cam),
        "rgb_archive_images": len(image_names),
        "rgb_usable_records": len(rows) - len(rgb_missing_rows),
        "rgb_missing_record_count": len(rgb_missing_rows),
        "rgb_missing_at_last_record_only": bool(rows) and rgb_missing_rows == [len(rows) - 1],
        "rgb_distinct_referenced": len(counts),
        "rgb_reused_record_count": sum(n - 1 for n in counts.values()),
        "rgb_files_missing": len(image_missing),
        "rgb_files_unreferenced": len(image_unreferenced),
        "rgb_reference_path_mismatches": path_mismatches,
        "world_zero_based_consecutive": indexed,
        "world_camera_marker_ids_match_all": marker_ids_match,
        "camera_frame_numbers_nondecreasing": camera_nondecreasing,
        "camera_frame_numbers_unique_contiguous": camera_unique_contiguous,
        "camera_with_drop_numbers_consecutive": wdrop_consecutive,
        "first_record": rows[0] if rows else None,
        "last_record": rows[-1] if rows else None,
        "knee_included_angle_comparison": angles,
        "record_index_alignment_supported": record_index_ok,
        "rgb_mapping_complete": rgb_mapping_ok,
        "complete_rgb_alignment_audit_pass": complete_audit_ok,
        "rgb_one_to_one_with_wcs": bool(rows) and len(counts) == len(world),
        "physical_timestamp_synchronization_verified": False,
        "angle_agreement_threshold_degrees": 1.0,
        "angle_agreement_is_independent_validation": False,
        "interpretation": (
            "World/camera 3D records are consistent by list index; RGB paths map "
            "via camera metadata, possibly many-to-one. No exposure-time or "
            "hardware-clock timestamps were tested; do not treat this as "
            "independent physical synchronization proof."
        ),
    }
    return report, rows


def save_mapping(rows: list[dict], destination: str | Path) -> Path:
    """Write derived frame metadata ONLY to an explicitly chosen local output."""
    destination = Path(destination)
    if not rows:
        raise ValueError("Cannot export empty alignment mapping")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination
