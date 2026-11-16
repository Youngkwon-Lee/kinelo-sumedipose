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


def calibration_wand_fit_residuals(root, subject="S1", action="A3", repeat="D1",
                                   cameras=("C1", "C2", "C3", "C4", "C5", "C6")):
    """Compare source-provided wand projections against manually marked LED points.

    Crucially, the marked points may have been reused to FIT the projection.
    These residuals are not holdout reprojection accuracy and cannot validate
    hardware synchronization or independent camera calibration.
    """
    import json
    from zipfile import ZipFile

    root = Path(root)
    annotations = json.loads((root / "markers_points.json").read_text(encoding="utf-8"))
    subject_marks = annotations[subject]
    sample_member = f"{subject}{action}{repeat}.json"
    results = []
    for camera in cameras:
        annotation = subject_marks[camera]
        reference = np.asarray(annotation["points"], dtype=float)
        archive = root / "internal_data" / camera / f"{subject}.zip"
        with ZipFile(archive) as source:
            records = json.loads(source.read(sample_member))
        if not records:
            raise ValueError(f"Empty camera coordinate records: {camera}")
        projected = np.asarray(records[0]["wand_xy"], dtype=float)
        last = np.asarray(records[-1]["wand_xy"], dtype=float)
        if reference.shape != (5, 2) or projected.shape != reference.shape:
            raise ValueError(f"Expected five 2D wand correspondences: {camera}")
        if not np.isfinite(reference).all() or not np.isfinite(projected).all():
            raise ValueError(f"Nonfinite calibration wand point: {camera}")
        residual = projected - reference
        reference_path = Path(annotation.get("path", ""))
        relative_ok = not reference_path.is_absolute() and ".." not in reference_path.parts
        image = root / reference_path if relative_ok else None
        results.append({
            "camera": camera,
            "wand_point_count": len(reference),
            "source_annotation_fit_rmse_pixels": round(
                float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))), 4
            ),
            "source_annotation_fit_mean_abs_pixels": round(
                float(np.mean(np.linalg.norm(residual, axis=1))), 4
            ),
            "mean_projected_minus_annotated_x_pixels": round(float(np.mean(residual[:, 0])), 4),
            "mean_projected_minus_annotated_y_pixels": round(float(np.mean(residual[:, 1])), 4),
            "wand_projection_constant_within_recording": bool(
                np.allclose(last, projected, atol=1e-8)
            ),
            "annotated_reference_jpg_present_as_standalone_file": bool(
                image and image.is_file()
            ),
            "out_of_sample_accuracy_proven": False,
        })
    return {
        "subject": subject,
        "action": action,
        "repeat": repeat,
        "source": "markers_points.json manual annotations vs internal_data wand_xy",
        "wand_fit_residuals": results,
        "in_sample_fit_caveat": (
            "Manual LED annotations may have been used to calibrate the camera "
            "projection; this comparison is NOT an independent holdout reprojection test."
        ),
        "camera_to_vicon_exposure_timestamps_verified": False,
        "independent_calibration_accuracy_verified": False,
    }
