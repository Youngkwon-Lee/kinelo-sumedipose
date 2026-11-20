"""Synthetic-only S17 source CSV role and PnP sensitivity regression tests."""
from copy import deepcopy
from pathlib import Path
import csv

import numpy as np
import pytest

from sumedipose.c6_source_provenance_sensitivity import (
    source_role_candidates, reproduce_original_group_from_alt,
    five_group_sensitivity, TARGET_GROUP, CAMERA,
)
from sumedipose.position_holdout import ANCHORS, VARIANTS


def _camera_K(fx):
    return np.array([[fx,0.,575.],[0.,fx,900.],[0.,0.,1.]])


def test_S17_original_C6_row_may_explicitly_reference_S23_file(tmp_path):
    root = tmp_path
    folder = root/"calibration"/"external"/"matrix"
    folder.mkdir(parents=True)
    paths = (
        ("6","SS23_G65_main_row.npz",1550.),
        ("5","SS17_G65_paired_row.npz",1520.),
    )
    for camera_role,name,fx in paths:
        np.savez(folder/name,mtx_A=_camera_K(fx),
                 dist_A=np.zeros((1,5)))
    with (root/"calibration"/"external_calibration.csv").open(
            "w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=[
            "subject","cam","main","main_link","path"])
        writer.writeheader()
        for cam,name,fx in paths:
            writer.writerow({
                "subject":"S17","cam":cam,"main":"6","main_link":"True",
                "path":f"calibration/external/matrix/{name}"})
    roles=source_role_candidates(root)
    assert roles["6"][0][0,0]==1550.
    assert roles["5"][0][0,0]==1520.
    assert roles["6"][2]["filename_matches_declared_S17_group"] is False
    assert roles["5"][2]["filename_matches_declared_S17_group"] is True


def test_missing_or_ambiguous_source_camera_role_fails_closed(tmp_path):
    root=tmp_path
    folder=root/"calibration"/"external"/"matrix"
    folder.mkdir(parents=True)
    np.savez(folder/"SS23_G65_only.npz", mtx_A=_camera_K(1500.), dist_A=np.zeros((1,5)))
    path=root/"calibration"/"external_calibration.csv"
    with path.open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=[
            "subject","cam","main","main_link","path"])
        writer.writeheader()
        writer.writerow({
            "subject":"S17","cam":"6","main":"6","main_link":"True",
            "path":"calibration/external/matrix/SS23_G65_only.npz"})
    with pytest.raises(ValueError,match="source CSV paths for camera role"):
        source_role_candidates(root)


def _metric(sixjoint=30.,root=40.,project=10.):
    return {
        "n_people":1,
        "valid_source_frames":12,
        "valid_PnP_frames":12,
        "group_mean_root_error_pct_femur":root,
        "group_mean_sixjoint_error_pct_femur":sixjoint,
        "median_model_self_reprojection_px":9.6,
        "camera_reference_projection_p90_px":project,
    }


def test_previous_PnP_fold_must_reproduce_exactly():
    original=_metric()
    assert reproduce_original_group_from_alt(deepcopy(original),original)
    drift=deepcopy(original)
    drift["group_mean_sixjoint_error_pct_femur"]=31.
    with pytest.raises(ValueError,match="not reproduced"):
        reproduce_original_group_from_alt(drift,original)


def test_only_one_S17_group_is_substituted_and_other_four_unchanged():
    reports=[]
    for cam in ("C1","C6"):
        for model in VARIANTS:
            for group in ANCHORS:
                reports.append({
                    "camera":cam,"model":model,"heldout_group":group,
                    "A3":_metric(sixjoint=10.+ANCHORS.index(group)),
                    "A4":_metric(sixjoint=20.+ANCHORS.index(group)),
                })
    previous={"rows":reports}
    corrected={m:{
        "A3":_metric(sixjoint=40.,root=80.,project=3.),
        "A4":_metric(sixjoint=50.,root=90.,project=4.)
    } for m in VARIANTS}
    cells=five_group_sensitivity(previous,corrected)
    assert len(cells)==4
    for cell in cells:
        a=cell["aggregate_comparison"]["previous_group_filename_rule"]
        b=cell["aggregate_comparison"]["source_explicit_main_camera_row"]
        assert a["group_equal_median_sixjoint_pct_femur"] == (
            12. if cell["action"]=="A3" else 22.)
        assert b["group_equal_median_sixjoint_pct_femur"] == (
            12. if cell["action"]=="A3" else 22.)
        assert cell["S17_source_role_projection_p90_px"] in (3.,4.)


def test_exact_twenty_fold_identities_required():
    rows=[{"camera":c,"model":m,"heldout_group":g,
           "A3":_metric(),"A4":_metric()}
          for c in ("C1","C6") for m in VARIANTS for g in ANCHORS]
    rows[0]=deepcopy(rows[1])
    with pytest.raises(ValueError,match="Original five-group PnP report identities"):
        five_group_sensitivity({"rows":rows}, {
            model:{"A3":_metric(),"A4":_metric()} for model in VARIANTS})
