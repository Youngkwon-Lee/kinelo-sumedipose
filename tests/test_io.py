import json
from zipfile import ZipFile

from sumedipose.io import sample_paths, read_json_member, record_summary


def test_expected_zip_layout():
    p = sample_paths("raw", "C1", "S1", "A3", "D1")
    assert str(p["rgb"]).replace("\\", "/").endswith("frames/C1/S1/C1S1A3D1.zip")
    assert p["world_member"] == "S1A3D1.json"


def test_reads_without_extracting(tmp_path):
    archive = tmp_path / "x.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("record.json", json.dumps([{"frame_num": 0}]))
    assert read_json_member(archive, "record.json") == [{"frame_num": 0}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.zip"]


def test_summary_does_not_claim_sync():
    w = [{"frame_num": 0, "point_ids": ["LHJC"]}]
    c = [{"frame_num": 277, "point_ids": ["LHJC"]}]
    result = record_summary(w, c)
    assert result["marker_id_match_at_first_record"] is True
    assert result["temporal_alignment_verified"] is False
