"""Exploratory RGB optical-flow versus annotated marker motion audit.

Never exports images; marker-selected ROI is not independent pose localization.
No exposure timestamps or Vicon hardware-clock synchronization are tested.
"""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .io import read_sample


def lag_correlations(expected, observed, max_lag=4):
    a, b = np.asarray(expected, float), np.asarray(observed, float)
    if a.ndim != 1 or a.shape != b.shape or max_lag < 0:
        raise ValueError("Unequal/invalid motion series or lag")
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        x = a[lag:] if lag > 0 else a[:lag] if lag < 0 else a
        y = b[:-lag] if lag > 0 else b[-lag:] if lag < 0 else b
        good = np.isfinite(x) & np.isfinite(y)
        out[str(lag)] = (round(float(np.corrcoef(x[good], y[good])[0, 1]), 7)
                         if good.sum() >= 3 and np.std(x[good]) > 1e-8
                         and np.std(y[good]) > 1e-8 else None)
    return out


def optical_flow_audit(root, camera="C1", subject="S1", repeat="D1",
                       *, stride=6, image_width=288, max_lag=4,
                       marker_ids=("C7", "CLAV"), displacement_axis="y",
                       roi_half_width=32, roi_half_height=36):
    """Use actual JPG pixels to assess motion plausibility, not physical clock sync."""
    if not 1 <= stride <= 200 or not 128 <= image_width <= 1024 or not 0 <= max_lag <= 50:
        raise ValueError("Invalid analysis parameters")
    if not marker_ids or len(set(marker_ids)) != len(marker_ids) or not all(isinstance(x, str) and x for x in marker_ids):
        raise ValueError("Provide unique named tracking markers")
    if displacement_axis not in ("x", "y"):
        raise ValueError("displacement_axis must be x or y")
    if not 4 <= roi_half_width <= 512 or not 4 <= roi_half_height <= 512:
        raise ValueError("ROI dimensions out of range")
    axis = 0 if displacement_axis == "x" else 1
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install optional image extra: pip install -e '.[image]'") from exc
    paths, _, records = read_sample(root, camera, subject, "A3", repeat)
    groups = defaultdict(list)
    prefix = f"{camera}{subject}A3{repeat}_"
    for i, rec in enumerate(records):
        name = Path(str(rec.get("path", ""))).name
        if name.startswith(prefix) and name.endswith(".jpg"):
            groups[name].append(i)

    with ZipFile(paths["rgb"]) as z:
        available = set(z.namelist())
        absent = [name for name in groups if name not in available]
        entries = [(name, inds[len(inds)//2]) for name, inds in groups.items() if name in available]
        pairs = [(entries[i], entries[i+stride]) for i in range(0, len(entries)-stride, stride)]
        expected, observed, features = [], [], []

        def pixels(name):
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            h, w = img.shape
            oh = max(1, round(h*image_width/w))
            return cv2.resize(img, (image_width, oh), interpolation=cv2.INTER_AREA), (image_width/w, oh/h)

        def point(rec, scale):
            ids = rec["point_ids"]
            coord = np.asarray(rec["xy"], float)
            p = coord[[ids.index(m) for m in marker_ids]].mean(axis=0)*scale
            if not np.isfinite(p).all():
                raise ValueError("Non-finite projection")
            return p

        for (name_a, ia), (name_b, ib) in pairs:
            result_a, result_b = pixels(name_a), pixels(name_b)
            if result_a is None or result_b is None:
                continue
            a, scale_a = result_a
            b, scale_b = result_b
            if a.shape != b.shape:
                continue
            try:
                p, q = point(records[ia], scale_a), point(records[ib], scale_b)
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            mask = np.zeros_like(a)
            x, y = np.rint(p).astype(int)
            x0, x1 = max(0, x-roi_half_width), min(a.shape[1], x+roi_half_width)
            y0, y1 = max(0, y-roi_half_height), min(a.shape[0], y+roi_half_height)
            if x1 <= x0 or y1 <= y0:
                continue
            mask[y0:y1, x0:x1] = 255
            pts = cv2.goodFeaturesToTrack(a, 60, .005, 4, mask=mask)
            if pts is None:
                continue
            nxt, status, err = cv2.calcOpticalFlowPyrLK(
                a, b, pts, None, winSize=(29, 29), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            if nxt is None or status is None or err is None:
                continue
            rev, back_status, _ = cv2.calcOpticalFlowPyrLK(
                b, a, nxt, None, winSize=(29, 29), maxLevel=3)
            if rev is None or back_status is None:
                continue
            fb = np.linalg.norm((rev-pts).reshape(-1, 2), axis=1)
            good = ((status.ravel()==1) & (back_status.ravel()==1)
                    & (err.ravel()<25) & (fb<2))
            if good.sum() < 5:
                continue
            expected.append(float(q[axis]-p[axis]))
            observed.append(float(np.median((nxt-pts).reshape(-1,2)[good,axis])))
            features.append(int(good.sum()))

    lag = lag_correlations(expected, observed, max_lag)
    ranked = [(int(k),v) for k,v in lag.items() if v is not None]
    best = max(ranked,key=lambda pair:pair[1])[0] if ranked else None
    err = np.abs(np.asarray(expected)-np.asarray(observed))
    median_err = round(float(np.median(err)),4) if len(err) else None
    same = lag["0"]
    plausible = bool(len(observed)>=15 and same is not None and same>=.8
                     and best==0 and median_err is not None and median_err<3)
    first_n = len(expected) // 2
    half_corr = {
        "first_half": lag_correlations(expected[:first_n], observed[:first_n], 0)["0"],
        "second_half": lag_correlations(expected[first_n:], observed[first_n:], 0)["0"],
    }
    return {
        "sample": f"{camera}{subject}A3{repeat}",
        "method": "forward-backward LK optical flow of RGB upper torso",
        "roi_uses_camera_marker_projection": True,
        "tracking_marker_ids": list(marker_ids),
        "displacement_axis": displacement_axis,
        "roi_half_width_analysis_pixels": roi_half_width,
        "roi_half_height_analysis_pixels": roi_half_height,
        "stride_unique_rgb_images": stride,
        "analysis_image_width_pixels": image_width,
        "lag_unit": "sampled_pairs_not_frames_or_seconds",
        "camera_3d_records": len(records),
        "rgb_referenced_unique_members": len(groups),
        "rgb_existing_unique_members": len(entries),
        "rgb_missing_members_masked": len(absent),
        "attempted_motion_pairs": len(pairs),
        "valid_motion_pairs": len(observed),
        "median_tracked_features": round(float(np.median(features)),2) if features else None,
        "median_abs_displacement_error_downscaled_px": median_err,
        "motion_correlation_lag_zero": same,
        "best_lag_in_sampled_pair_units": best,
        "lag_correlations": lag,
        "lag_zero_correlation_by_half": half_corr,
        "exploratory_rgb_motion_consistency_supported": plausible,
        "physical_exposure_timestamp_sync_verified": False,
        "clinical_measurement_validated": False,
        "limitations": ("ROI is selected from WCS-derived camera markers, so not blind pose localization. "
                        "Observed pixel flow is image-derived; 3D model timestamps, exposure times, "
                        "hardware sync, clinical validity and model accuracy remain unverified. "
                        "No raw or tracked per-frame pixel data are saved.")
    }
