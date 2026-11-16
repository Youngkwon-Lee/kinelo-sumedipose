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
