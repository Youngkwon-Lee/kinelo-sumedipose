"""Synthetic-only Vicon joint-centre versus surface-marker reference sensitivity."""
import numpy as np
import pytest

from sumedipose.marker_reference_sensitivity import (
    ACTIONS, ALTERNATIVES, BANDS, CAMERAS, aggregate,
    named_marker_angle, reference_sensitivity,
)
from sumedipose.depth_angle_decomposition import GROUPS
from sumedipose.position_holdout import ANCHORS


def _record():
    ids=["LHJC","LKJC","LAJC","LKNE","LANK","RHJC","RKJC","RAJC","RKNE","RANK"]
    three=[
        [1,0,0],[0,0,0],[0,1,0],
        [0.3,0,0],[0,1.3,0],
        [1,0,0],[0,0,0],[0,1,0],
        [0.3,0,0],[0,1.3,0],
    ]
    return {"point_ids":ids,"xyz":three}


def test_joint_reference_marker_variations_are_not_ai_error():
    r=_record()
    base=named_marker_angle(r,("LHJC","LKJC","LAJC"))
    assert base==pytest.approx(90)
    diff=reference_sensitivity(r,"L")
    assert set(diff)==set(("band",*ALTERNATIVES))
    assert diff["knee_surface"]!=pytest.approx(0)
    assert diff["ankle_surface"]==pytest.approx(0)
    assert diff["knee_and_ankle_surface"]!=pytest.approx(0)
    assert reference_sensitivity(r,"R")==diff


def test_missing_source_marker_fails_closed():
    r=_record()
    r["point_ids"].remove("LKNE")
    r["xyz"].pop(3)
    with pytest.raises(ValueError,match="lacks"):
        reference_sensitivity(r,"L")
    with pytest.raises(ValueError,match="Side"):
        reference_sensitivity(r,"X")


def _row(subject,action,cam):
    metrics={"valid_knee_angles":48}
    for name in ALTERNATIVES:
        metrics[name+"_mean_absolute_deg"]=4.
        metrics[name+"_mean_signed_deg"]=2.
    return {
        "subject":subject,"calibration_group":GROUPS[subject],
        "action":action,"camera":cam,
        "strict_unique_jpeg_samples":24,
        "all":metrics,
        "by_phase":{band:metrics for band in BANDS},
    }


def test_all26_A3_five_A4_group_aggregates():
    rows=[_row(s,act,c) for act in ACTIONS
          for s in (tuple(GROUPS) if act=="A3" else ANCHORS)
          for c in CAMERAS]
    out=aggregate(rows)
    assert out["evaluated_recording_views"]==62
    assert out["unique_jpeg_instances"]==1488
    assert out["reference_side_angle_observations"]==2976
    assert len(out["results"])==16
    sample=next(x for x in out["results"] if
        x["action"]=="A3" and x["camera"]=="C6" and x["phase"]=="all")
    assert sample["groups_with_data"]==5
    assert sample["knee_surface_mean_absolute_deg_mean_of_group_means"]==pytest.approx(4)


def test_missing_Vicon_source_view_is_rejected():
    rows=[_row(s,act,c) for act in ACTIONS
          for s in (tuple(GROUPS) if act=="A3" else ANCHORS)
          for c in CAMERAS]
    with pytest.raises(ValueError,match="complete"):
        aggregate(rows[:-1])
