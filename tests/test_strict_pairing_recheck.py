"""Synthetic-only tests; no participant recordings, private coordinates or model assets."""
from copy import deepcopy
from pathlib import Path

import pytest

from sumedipose.strict_pairing_recheck import (
    ACTIONS, CAMERAS, FRAMES, MODALITIES, SESSIONS, VARIANTS,
    aggregate_strict, compare_original_replay, dataset_cohorts,
    _group_correlation,
)


def synthetic_protocol():
    groups = {}
    for offset, root, n in ((1, "S1", 4), (6, "S6", 4),
                            (11, "S11", 6), (17, "S17", 6), (23, "S23", 6)):
        for i in range(offset, offset + n):
            groups[f"S{i}"] = root
    census = {
        "missing_subjects": ["S5", "S10"],
        "calibration_group_sizes": {
            "S1": 4, "S6": 4, "S11": 6, "S17": 6, "S23": 6,
        },
    }
    new = {
        "subject_groups": [
            {"calibration_group": "S1", "subjects": ["S2","S3"]},
            {"calibration_group": "S6", "subjects": ["S7","S8"]},
            {"calibration_group": "S11", "subjects": ["S12","S13"]},
            {"calibration_group": "S17", "subjects": ["S18","S19"]},
            {"calibration_group": "S23", "subjects": ["S24","S25"]},
        ]
    }
    return census, new, groups


def fake_report(action, subj, cam, mod, group, old, new):
    def measures(value):
        item = {"valid": 24, "attempted": 24, "MAE_degrees": value}
        return {dim: {"left": dict(item), "right": dict(item)}
                for dim in MODALITIES}
    return {
        "sample": f"{cam}{subj}{action}D2",
        "model_variant": mod,
        "model_asset_sha256": f"hash-{mod}",
        "unique_images_sampled": FRAMES,
        "max_unique_images_requested": FRAMES,
        "model_detected_poses": FRAMES,
        "inference_mode": "IMAGE",
        "pairing": {
            "original_wrong_coordinate_references": int(cam == "C1"),
            "changed_jpg_name_count": 0,
            "frozen_same_rgb_frames": True,
        },
        "original_results": measures(old),
        "strict_results": measures(new),
    }


def inputs():
    census, newer, groupmap = synthetic_protocol()
    groups = dataset_cohorts(census, newer, groupmap)
    cal = {
        "comparisons": [
            {"session": s, "pair": "G65", "sampson_median_px": i}
            for i, s in enumerate(SESSIONS, 1)
        ]
    }
    reports = {}
    for act, members in (("A3", groups["subjects"]), ("A4", set(SESSIONS))):
        for sub in members:
            for cam in CAMERAS:
                for mod in VARIANTS:
                    idx = SESSIONS.index(groupmap[sub]) + 1
                    old = round(10 + idx + int(sub[1:]) / 1000, 4)
                    new = round(old + (0.02 if idx == 5 else -0.01), 4)
                    reports[(act, sub, cam, mod)] = fake_report(
                        act, sub, cam, mod, groupmap[sub], old, new)
    prior = {"per_outcome": []}
    for cam in CAMERAS:
        for mod in VARIANTS:
            for space in MODALITIES:
                c = {}
                for label, members in groups["cohorts"].items():
                    old = _group_correlation(
                        members, groupmap, reports,
                        {s: i for i, s in enumerate(SESSIONS, 1)},
                        action="A3", cam=cam, model=mod,
                        modality=space, version="original")
                    c[label] = {"rho_descriptive": old["rho"]}
                prior["per_outcome"].append({
                    "camera": cam, "model": mod,
                    "reference_space": space, "cohorts": c,
                })
    return census, newer, groupmap, cal, reports, prior


def test_disjoint_cohorts_and_exact_five_shared_calibration_groups():
    census, newer, map_ = synthetic_protocol()
    found = dataset_cohorts(census, newer, map_)
    assert len(found["subjects"]) == 26
    assert [len(found["cohorts"][c]) for c in (
        "anchor5_D2", "disjoint10_D2", "remaining11_D2", "all26_D2"
    )] == [5, 10, 11, 26]
    assert set(found["cohorts"]["anchor5_D2"]).isdisjoint(
        found["cohorts"]["disjoint10_D2"])


def test_complete_124_reports_40_associations_and_no_inflated_n():
    x = inputs()
    output = aggregate_strict(x[0], x[5], x[2], x[1], x[3], x[4])
    assert output["n_A3_participants"] == 26
    assert output["n_A4_anchor_participants"] == 5
    assert output["model_reports_checked"] == 124
    assert output["model_image_inferences"] == 2976
    assert output["n_group_descriptive_associations"] == 40
    assert output["independent_calibration_groups"] == 5
    assert output["new_independent_calibration_groups"] == 0
    assert output["total_old_wrong_pairings"] == 31  # synthetic one bad reference per C1 JPG, one model only
    assert all(x["delta_rho"] == 0 for x in output["comparison"])
    assert output["absolute_3d_MPJPE_mm_validated"] is False


def test_reject_missing_report_and_misleading_changed_images():
    census, newer, m, cal, reports, prior = inputs()
    reports.pop(("A3", "S4", "C1", "lite"))
    with pytest.raises(ValueError, match="Require 104"):
        aggregate_strict(census, prior, m, newer, cal, reports)
    census, newer, m, cal, reports, prior = inputs()
    reports[("A4", "S1", "C6", "full")]["pairing"]["changed_jpg_name_count"] = 1
    with pytest.raises(ValueError, match="Original image selection"):
        aggregate_strict(census, prior, m, newer, cal, reports)


def test_original_correlations_checked_against_old_frozen_protocol():
    census, newer, m, cal, reports, prior = inputs()
    prior["per_outcome"][0]["cohorts"]["all26_D2"]["rho_descriptive"] = -0.8
    with pytest.raises(ValueError, match="Old cohort association"):
        aggregate_strict(census, prior, m, newer, cal, reports)


def test_reproduced_unchanged_legacy_metrics_required():
    r = fake_report("A3", "S1", "C1", "lite", "S1", 12., 12.2)
    old = {
        "sample": r["sample"], "model_variant": r["model_variant"],
        "model_asset_sha256": r["model_asset_sha256"],
        "model_detected_poses": 24,
        "results": deepcopy(r["original_results"]),
    }
    assert compare_original_replay(r, old)
    old["results"]["world_3d"]["left"]["MAE_degrees"] = 15.
    with pytest.raises(ValueError, match="not reproduced"):
        compare_original_replay(r, old)


def test_group_counts_reject_unexpected_calibration_session():
    census, newer, map_ = synthetic_protocol()
    map_["S4"] = "OTHER"
    with pytest.raises(ValueError, match="26-person frozen cohorts"):
        dataset_cohorts(census, newer, map_)
