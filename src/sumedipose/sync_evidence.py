"""Evidence inventory for A3 timestamp data, without claiming clock synchronization."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

from .io import read_sample


def _timestamp_fields(record: dict) -> list[str]:
    exact = {"time_ms", "time_us", "timestamp", "timestamps", "capture_time",
             "exposure_time", "unix_time", "epoch_time", "datetime"}
    return sorted(str(key) for key in record
                  if (str(key).lower() in exact
                      or "timestamp" in str(key).lower()))


def timestamp_evidence_inventory(root, camera="C6", subject="S1", repeat="D1"):
    """Look for actual timestamp evidence in sample records and up to 3 JPGs.

    Index fields and ZIP modification dates are NOT treated as timestamps.
    No images or source coordinates are copied or returned.
    """
    root = Path(root)
    paths, world, cam = read_sample(root, camera, subject, "A3", repeat)
    world_fields = _timestamp_fields(world[0]) if world else []
    camera_fields = _timestamp_fields(cam[0]) if cam else []
    csv_fields = []
    if (root / "SUMediPose.csv").is_file():
        with (root / "SUMediPose.csv").open(encoding="utf-8-sig", newline="") as f:
            csv_fields = _timestamp_fields({key: None for key in next(csv.reader(f), [])})
    exif_checked, exif_time_count = 0, 0
    try:
        from PIL import Image, ExifTags
    except ImportError:
        Image = None

    with ZipFile(paths["rgb"]) as archive:
        members = set(archive.namelist())
        names = list(dict.fromkeys(Path(str(r.get("path", ""))).name for r in cam))
        names = [name for name in names if name in members]
        picks = sorted(set((0, len(names)//2, len(names)-1))) if names else []
        if Image is not None:
            tags = {"DateTime", "DateTimeOriginal", "DateTimeDigitized",
                    "OffsetTime", "OffsetTimeOriginal", "SubsecTimeOriginal"}
            for i in picks:
                with Image.open(io.BytesIO(archive.read(names[i]))) as image:
                    values = image.getexif()
                exif_checked += 1
                if any(ExifTags.TAGS.get(k) in tags for k in values):
                    exif_time_count += 1

    manual_wand_img_found = None
    annotations_path = root / "markers_points.json"
    if annotations_path.is_file():
        ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        entry = ann.get(subject, {}).get(camera)
        if entry:
            image = root / str(entry.get("path", ""))
            manual_wand_img_found = image.is_file()
            # Do not treat the calibration camera images as event timestamps.

    return {
        "sample": f"{camera}{subject}A3{repeat}",
        "wcs_timestamp_fields": world_fields,
        "camera_timestamp_fields": camera_fields,
        "dataset_csv_timestamp_columns": csv_fields,
        "jpeg_exif_images_examined": exif_checked,
        "jpeg_exif_images_with_datetime": exif_time_count,
        "calibration_wand_reference_image_present_as_file": manual_wand_img_found,
        "archive_zip_dates_used_as_capture_times": False,
        "frame_num_or_frame_num_w_drop_used_as_capture_times": False,
        "source_evidence_sufficient_for_independent_clock_sync": False,
        "independent_exposure_time_sync_verified": False,
        "notes": (
            "No direct common-clock exposure/Vicon timebase was verified. "
            "Frame numbers and ZIP mtime are not exposure timestamps. "
            "The source paper describes post hoc stomp alignment."
        ),
    }
