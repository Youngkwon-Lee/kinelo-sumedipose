"""Read-only inventory of source-provided external camera calibration metadata.

This verifies file structure only; the source's error units and independent
camera reprojection accuracy are NOT established.
"""
import csv
from pathlib import Path
import numpy as np


def external_calibration_inventory(root, subject="S1"):
    root = Path(root)
    csv_path = root / "calibration" / "external_calibration.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("subject") == subject]
    result = []
    for rec in sorted(rows, key=lambda item: int(item["cam"])):
        relative = Path(rec.get("path", "").replace("\\", "/"))
        valid_path = (not relative.is_absolute() and ".." not in relative.parts
                      and relative.parts[:3] == ("calibration", "external", "matrix"))
        matrix_path = root / relative if valid_path else None
        exists = bool(matrix_path and matrix_path.is_file())
        shapes_ok = False
        if exists:
            try:
                with np.load(matrix_path, allow_pickle=False) as data:
                    shapes_ok = all(
                        key in data.files and np.asarray(data[key]).shape == shape
                        and np.isfinite(data[key]).all()
                        for key, shape in {"R": (3, 3), "T": (3, 1),
                                           "mtx_A": (3, 3), "mtx_B": (3, 3)}.items()
                    )
            except (ValueError, OSError, KeyError):
                shapes_ok = False
        result.append({
            "camera": "C" + rec["cam"],
            "paired_camera": "C" + rec["pair"],
            "calibration_frames_reported": int(rec["num_frames"]),
            "source_error_value": float(rec["error"]),
            "source_error_skel_value": float(rec["error_skel"]),
            "error_unit_verified": False,
            "calibration_matrix_file_present": exists,
            "matrix_shapes_finite": shapes_ok,
            "reprojection_to_original_rgb_independently_validated": False,
        })
    return {"subject_code": subject, "camera_pairs": result,
            "all_structurally_readable": bool(result) and all(
                row["calibration_matrix_file_present"] and row["matrix_shapes_finite"]
                for row in result),
            "external_camera_calibration_accuracy_validated": False}
