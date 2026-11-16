"""Published calibration-board nominal scale; not independent metrology.

SUMediPose paper, section 5.3.2 "Camera system calibration":
An A2 board with 60 mm square lengths is used for all camera calibrations.
The published external calibration code defines world_scaling=1 per square.
This author-reported nominal conversion is NOT independent Vicon 3D accuracy.
"""
from __future__ import annotations

import math

PUBLISHED_NOMINAL_SQUARE_MM = 60.0
SOURCE_EXTRINSIC_WORLD_SCALING = 1.0
BOARD_SIZE_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12088771/"
BOARD_SCALE_CODE_URL = "https://github.com/chris-schreuder/SUMediPose_pipeline/blob/main/code/calibration/extrinsic.py"


def board_square_units_to_nominal_mm(value: float) -> float:
    """Convert units relative to the published 60-mm board, not measured error."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Board square measurement must be finite")
    return value * PUBLISHED_NOMINAL_SQUARE_MM / SOURCE_EXTRINSIC_WORLD_SCALING


def published_board_scale_evidence() -> dict:
    """Machine-readable source and qualification for the nominal conversion."""
    return {
        "author_reported_nominal_square_mm": PUBLISHED_NOMINAL_SQUARE_MM,
        "source_extrinsic_world_scaling": SOURCE_EXTRINSIC_WORLD_SCALING,
        "equivalent_nominal_mm_per_source_square_unit": (
            PUBLISHED_NOMINAL_SQUARE_MM / SOURCE_EXTRINSIC_WORLD_SCALING
        ),
        "source_paper": BOARD_SIZE_PAPER_URL,
        "source_calibration_code": BOARD_SCALE_CODE_URL,
        "independent_physical_board_measurement_verified": False,
        "vicon_world_absolute_3d_accuracy_verified": False,
        "notes": (
            "The published 60-mm square establishes a nominal physical "
            "calibration scale. It does not establish actual camera-to-Vicon "
            "registration, absolute 3D reconstruction accuracy or clock sync."
        ),
    }
