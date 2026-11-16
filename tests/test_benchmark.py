"""Only synthetic data; no real RGB or subject features used."""
import math
from sumedipose.benchmark import JOINT_NAMES, unique_sample_indices, summarize_errors, _angles_from_named


def test_unique_rgb_sampling_is_deterministic_and_drops_missing():
    records = [
        {"path":"frames/C1/S1/C1S1A3D1/C1S1A3D1_0001.jpg"},
        {"path":"frames/C1/S1/C1S1A3D1/C1S1A3D1_0001.jpg"},
        {"path":"frames/C1/S1/C1S1A3D1/C1S1A3D1_0002.jpg"},
        {"path":"frames/C1/S1/C1S1A3D1/C1S1A3D1_0003.jpg"},
        {"path":"frames/C1/S1/C1S1A3D1/C1S1A3D1_0003.jpg"},
    ]
    selected, missing = unique_sample_indices(records,["C1S1A3D1_0001.jpg","C1S1A3D1_0003.jpg"],max_frames=4)
    assert selected == [("C1S1A3D1_0001.jpg",1),("C1S1A3D1_0003.jpg",4)]
    assert missing == 1


def test_unique_rgb_sampling_respects_max_frames():
    records=[{"path":f"x/{i}.jpg"} for i in range(20)]
    selected,missing=unique_sample_indices(records,[f"{i}.jpg" for i in range(20)],max_frames=4)
    assert len(selected)==4
    assert selected[0][0]=="0.jpg"
    assert selected[-1][0]=="19.jpg"
    assert missing==0


def test_angle_reference_2d_and_3d_use_named_centres():
    r={"point_ids": ["LHJC","LKJC","LAJC"],
       "xy": [[1,0],[0,0],[0,1]],
       "xyz": [[1,0,0],[0,0,0],[0,1,0]]}
    assert _angles_from_named(r,JOINT_NAMES["left"]["reference"],"xy")==90.0
    assert _angles_from_named(r,JOINT_NAMES["left"]["reference"],"xyz")==90.0


def test_summary_error_handling():
    m=summarize_errors([3,4,math.nan],attempted=4)
    assert m["valid"]==2 and m["attempted"]==4
    assert m["MAE_degrees"]==3.5
    assert abs(m["RMSE_degrees"]-math.sqrt(12.5))<0.0001
    assert m["fraction_abs_error_over_20_degrees"]==0
    empty=summarize_errors([math.nan],attempted=2)
    assert empty["valid"]==0 and empty["MAE_degrees"] is None
