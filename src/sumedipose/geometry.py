"""Simple geometry; input units do not affect the resulting angle."""
from __future__ import annotations

import numpy as np


def joint_angle(proximal, joint, distal) -> float:
    """Return the included angle at joint in degrees; NaN for degenerate vectors."""
    a = np.asarray(proximal, dtype=float) - np.asarray(joint, dtype=float)
    b = np.asarray(distal, dtype=float) - np.asarray(joint, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norm) or norm <= 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1.0, 1.0))))


def angle_from_markers(record: dict, proximal_id: str, joint_id: str, distal_id: str) -> float:
    """Compute angle from the named 3D coordinates in a single WCS frame."""
    ids = record["point_ids"]
    pts = record["xyz"]
    return joint_angle(*(pts[ids.index(name)] for name in (proximal_id, joint_id, distal_id)))


def knee_angles(record: dict) -> dict[str, float]:
    """Anatomical included angles from hip/knee/ankle joint centers."""
    return {
        "left_knee_degrees": angle_from_markers(record, "LHJC", "LKJC", "LAJC"),
        "right_knee_degrees": angle_from_markers(record, "RHJC", "RKJC", "RAJC"),
    }
