"""Read original SUMediPose A3 archives without extracting or modifying them."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile


def sample_paths(root: str | Path, camera: str = "C1", subject: str = "S1", action: str = "A3", repeat: str = "D1") -> dict[str, Path | str]:
    """Return paths to original A3 files; no path creation or mutation."""
    root = Path(root)
    key = f"{subject}{action}{repeat}"
    return {
        "rgb": root / "frames" / camera / subject / f"{camera}{key}.zip",
        "world": root / "WCS" / f"{subject}.zip",
        "camera": root / "internal_data" / camera / f"{subject}.zip",
        "world_member": key + ".json",
    }


def read_json_member(archive: str | Path, member: str):
    """Load one JSON member from a ZIP safely, without extracting any files."""
    with ZipFile(archive) as z:
        return json.loads(z.read(member))


def read_sample(root: str | Path, camera="C1", subject="S1", action="A3", repeat="D1"):
    """Return three metadata objects; avoid copying large RGB image arrays."""
    paths = sample_paths(root, camera, subject, action, repeat)
    name = str(paths["world_member"])
    world = read_json_member(paths["world"], name)
    camera_records = read_json_member(paths["camera"], name)
    if not isinstance(world, list) or not isinstance(camera_records, list):
        raise ValueError("Expected lists of frame records")
    return paths, world, camera_records


def record_summary(world: list, camera_records: list) -> dict:
    """Summarize a sample without assuming time alignment or spatial units."""
    marker_match = bool(world and camera_records and world[0].get("point_ids") == camera_records[0].get("point_ids"))
    return {
        "wcs_records": len(world),
        "camera_records": len(camera_records),
        "marker_id_match_at_first_record": marker_match,
        "marker_count": len(world[0].get("point_ids", [])) if world else 0,
        "first_wcs_frame_num": world[0].get("frame_num") if world else None,
        "first_camera_frame_num": camera_records[0].get("frame_num") if camera_records else None,
        "temporal_alignment_verified": False,
    }
