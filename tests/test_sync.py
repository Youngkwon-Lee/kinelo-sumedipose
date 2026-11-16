"""Synthetic tests only; never need any real participant files."""
import csv
import json
import math
from zipfile import ZipFile

from sumedipose.sync import audit_sample, save_mapping


def _sample(tmp_path, *, image_ids=(100, 101, 102), shift=False, corrupt_path=False):
    root = tmp_path / "raw"
    rgb = root / "frames" / "C1" / "S1" / "C1S1A3D1.zip"
    worldzip = root / "WCS" / "S1.zip"
    camzip = root / "internal_data" / "C1" / "S1.zip"
    for p in (rgb, worldzip, camzip):
        p.parent.mkdir(parents=True, exist_ok=True)
    marker_ids = ["LHJC", "LKJC", "LAJC", "RHJC", "RKJC", "RAJC"]
    angles = (90, 110, 75, 130)
    def xyz(degrees):
        a = math.radians(degrees)
        return [[-1, 0, 0], [0, 0, 0], [math.cos(a), math.sin(a), 0],
                [1, 0, 0], [0, 0, 0], [math.cos(a), math.sin(a), 0]]
    world, camera = [], []
    record_images = (100, 100, 101, 102)
    for index, im in enumerate(record_images):
        p = f"frames/C1/S1/C1S1A3D1/C1S1A3D1_{im:04d}.jpg"
        world.append({"frame_num": index, "point_ids": marker_ids, "xyz": xyz(angles[index])})
        camera.append({
            "frame_num": im,
            "frame_num_w_drop": 20 + index,
            "path": ("frames/unknown/" + p) if corrupt_path and index == 0 else p,
            "point_ids": marker_ids,
            "xyz": [[x * .03 for x in v] for v in xyz(angles[(index + 1) % 4] if shift else angles[index])],
        })
    with ZipFile(rgb, "w") as f:
        for n in image_ids:
            f.writestr(f"C1S1A3D1_{n:04d}.jpg", b"synthetic")
    with ZipFile(worldzip, "w") as f:
        f.writestr("S1A3D1.json", json.dumps(world))
    with ZipFile(camzip, "w") as f:
        f.writestr("S1A3D1.json", json.dumps(camera))
    return root


def test_one_index_to_many_rgb_is_not_forced_one_to_one(tmp_path):
    root = _sample(tmp_path)
    result, rows = audit_sample(root)
    assert result["wcs_records"] == 4
    assert result["camera_records"] == 4
    assert result["rgb_archive_images"] == 3
    assert result["rgb_distinct_referenced"] == 3
    assert result["rgb_reused_record_count"] == 1
    assert result["world_camera_marker_ids_match_all"] is True
    assert result["rgb_mapping_complete"] is True
    assert result["record_index_alignment_supported"] is True
    assert result["rgb_one_to_one_with_wcs"] is False
    assert result["physical_timestamp_synchronization_verified"] is False
    assert result["complete_rgb_alignment_audit_pass"] is True
    assert rows[0]["rgb_member"] == rows[1]["rgb_member"]
    assert rows[0]["wcs_frame_num"] == 0
    assert rows[0]["camera_frame_num"] == 100


def test_angle_mismatch_rejects_index_consistency(tmp_path):
    result, _ = audit_sample(_sample(tmp_path, shift=True))
    assert result["record_index_alignment_supported"] is False
    assert result["knee_included_angle_comparison"]["left"]["mean_abs_diff_degrees"] > 1


def test_missing_zip_member_is_detected(tmp_path):
    result, _ = audit_sample(_sample(tmp_path, image_ids=(100, 101)))
    assert result["rgb_files_missing"] == 1
    assert result["rgb_mapping_complete"] is False
    assert result["record_index_alignment_supported"] is True
    assert result["complete_rgb_alignment_audit_pass"] is False


def test_invalid_relative_path_is_detected(tmp_path):
    result, _ = audit_sample(_sample(tmp_path, corrupt_path=True))
    assert result["rgb_reference_path_mismatches"] == 1
    assert result["record_index_alignment_supported"] is True
    assert result["complete_rgb_alignment_audit_pass"] is False


def test_csv_export_only_requested_destination(tmp_path):
    root = _sample(tmp_path)
    _, rows = audit_sample(root)
    path = tmp_path / "analysis" / "mapping.csv"
    save_mapping(rows, path)
    with path.open(newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert len(loaded) == 4
    assert loaded[0]["wcs_frame_num"] == "0"
    assert (root / "WCS" / "S1.zip").exists()
