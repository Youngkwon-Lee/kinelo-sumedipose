"""All sample images are generated synthetic arrays; no human source data."""
import csv
import json
import math
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from sumedipose.temporal import lag_correlations, optical_flow_audit
from sumedipose.calibration import external_calibration_inventory


def _synthetic_case(tmp_path, terminal_missing=False):
    raw = tmp_path / "raw"
    rgb = raw / "frames/C1/S1/C1S1A3D1.zip"
    camera = raw / "internal_data/C1/S1.zip"
    wcs = raw / "WCS/S1.zip"
    for p in (rgb, camera, wcs):
        p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    patch = rng.integers(20, 240, (90, 92), dtype=np.uint8)
    recs, world = [], []
    with ZipFile(rgb, "w") as f:
        for i in range(37):
            center_y = 190 + round(22 * math.sin(i * 0.35))
            name = f"C1S1A3D1_{i:04d}.jpg"
            frame = np.full((450, 288), 45, np.uint8)
            frame[center_y-45:center_y+45, 98:190] = patch
            if i < 36 or not terminal_missing:
                ok, jpg = cv2.imencode(".jpg", frame)
                assert ok
                f.writestr(name, jpg.tobytes())
            recs.append({
                "frame_num": i, "point_ids": ["C7", "CLAV"],
                "path": f"frames/C1/S1/C1S1A3D1/{name}",
                "xy": [[144, center_y-10], [144, center_y+10]],
            })
            world.append({"frame_num": i, "point_ids": ["C7", "CLAV"], "xyz": [[0, 0, i],[0, 1, i]]})
    with ZipFile(camera, "w") as f:
        f.writestr("S1A3D1.json", json.dumps(recs))
    with ZipFile(wcs, "w") as f:
        f.writestr("S1A3D1.json", json.dumps(world))
    return raw


def test_zero_lag_correct_for_aligned_variable_motion():
    a = [math.sin(i * 0.8) for i in range(25)]
    r = lag_correlations(a, a)
    assert r["0"] == 1.0
    assert max(r, key=lambda k: r[k]) == "0"


def test_lag_inputs_checked_and_static_inconclusive():
    with pytest.raises(ValueError):
        lag_correlations([1, 2], [1])
    assert lag_correlations([0]*10, [0]*10)["0"] is None


def test_synthetic_images_verify_motion_but_not_hardware_sync(tmp_path):
    record = optical_flow_audit(_synthetic_case(tmp_path), stride=2)
    assert record["rgb_missing_members_masked"] == 0
    assert record["valid_motion_pairs"] >= 15
    assert record["motion_correlation_lag_zero"] > 0.92
    assert record["best_lag_in_sampled_pair_units"] == 0
    assert record["exploratory_rgb_motion_consistency_supported"] is True
    assert record["physical_exposure_timestamp_sync_verified"] is False


def test_synthetic_missing_last_image_is_masked(tmp_path):
    record = optical_flow_audit(_synthetic_case(tmp_path, terminal_missing=True), stride=2)
    assert record["rgb_missing_members_masked"] == 1
    assert record["rgb_existing_unique_members"] == 36
    assert record["physical_exposure_timestamp_sync_verified"] is False


def test_calibration_matrix_read_only_inventory(tmp_path):
    raw = tmp_path / "raw"
    d = raw / "calibration" / "external" / "matrix"
    d.mkdir(parents=True)
    file = d / "fixture.npz"
    np.savez(file, R=np.eye(3), T=np.zeros((3,1)), mtx_A=np.eye(3), mtx_B=np.eye(3))
    with (raw/"calibration"/"external_calibration.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject","cam","pair","num_frames","error","error_skel","path"])
        writer.writeheader()
        writer.writerow({"subject":"S1","cam":"6","pair":"5","num_frames":5,
                         "error":12.5,"error_skel":14.5,
                         "path":"calibration/external/matrix/fixture.npz"})
    result = external_calibration_inventory(raw)
    assert result["all_structurally_readable"] is True
    assert result["external_camera_calibration_accuracy_validated"] is False
    assert result["camera_pairs"][0]["source_error_value"] == 12.5
    assert file.exists()


def test_roi_marker_and_size_are_explicit(tmp_path):
    root = _synthetic_case(tmp_path)
    result = optical_flow_audit(
        root, stride=2, marker_ids=("C7",), roi_half_width=16,
        roi_half_height=18, displacement_axis="y",
    )
    assert result["tracking_marker_ids"] == ["C7"]
    assert result["roi_half_width_analysis_pixels"] == 16
    assert result["roi_half_height_analysis_pixels"] == 18
    assert result["displacement_axis"] == "y"
    assert result["physical_exposure_timestamp_sync_verified"] is False
    with pytest.raises(ValueError):
        optical_flow_audit(root, roi_half_width=2)
    with pytest.raises(ValueError):
        optical_flow_audit(root, marker_ids=("C7", "C7"))
    with pytest.raises(ValueError):
        optical_flow_audit(root, displacement_axis="z")


def test_wand_annotation_residual_is_not_holdout_accuracy(tmp_path):
    from sumedipose.calibration import calibration_wand_fit_residuals
    root = _synthetic_case(tmp_path)
    archive = root / "internal_data/C1/S1.zip"
    with ZipFile(archive) as z:
        records = json.loads(z.read("S1A3D1.json"))
    projections = [[i + 100.0, i + 200.0] for i in range(5)]
    for row in records:
        row["wand_xy"] = projections
    with ZipFile(archive, "w") as z:
        z.writestr("S1A3D1.json", json.dumps(records))
    annotations = {"S1": {"C1": {
        "points": [[x - 1.0, y] for x, y in projections],
        "path": "frames/C1/S1/C1S1A0D1/C1S1A0D1_0010.jpg",
    }}}
    (root / "markers_points.json").write_text(json.dumps(annotations), encoding="utf-8")
    report = calibration_wand_fit_residuals(root, cameras=("C1",))
    row = report["wand_fit_residuals"][0]
    assert row["wand_point_count"] == 5
    assert row["source_annotation_fit_rmse_pixels"] == 1.0
    assert row["wand_projection_constant_within_recording"] is True
    assert row["out_of_sample_accuracy_proven"] is False
    assert report["camera_to_vicon_exposure_timestamps_verified"] is False


def test_capture_time_evidence_does_not_confuse_indices_with_clock(tmp_path):
    from sumedipose.sync_evidence import timestamp_evidence_inventory
    root = _synthetic_case(tmp_path)
    report = timestamp_evidence_inventory(root, camera="C1")
    assert report["wcs_timestamp_fields"] == []
    assert report["camera_timestamp_fields"] == []
    assert report["jpeg_exif_images_with_datetime"] == 0
    assert report["frame_num_or_frame_num_w_drop_used_as_capture_times"] is False
    assert report["independent_exposure_time_sync_verified"] is False
