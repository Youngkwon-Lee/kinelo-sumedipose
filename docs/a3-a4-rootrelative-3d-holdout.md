# A3-trained → A4-held-out root-relative 3D joint geometry

**Study status:** experimental diagnostic only. This is **not MPJPE in true registered physical space**, does not measure absolute 3D joint position, and cannot be used as a medical accuracy endpoint.

## Ground truth and source pairing

The SUMediPose dataset offers Vicon WCS `xyz` for joint centres (`LHJC/LKJC/LAJC` and right equivalents). The `internal_data` camera XYZ is a transformation of that same Vicon reference, *not* another independent ground truth. The original paper explains that `dup_num != factor` means a dropped source camera frame may reuse a later JPEG filename even though the Vicon coordinate index does **not** correspond to that image. See [SUMediPose data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088771/).

Unlike the previous baseline's middle-record index selection, the new `strict_unique_pairs` must find **exactly one** metadata record for a real source JPEG where `dup_num == factor`; otherwise the JPEG is excluded. It checks original ZIP member existence and samples 24 **unique verified** image paths in original order. All five historical session anchors (S1/S6/S11/S17/S23), both actions (A3 squat / A4 deadlift) at D2, views C1/C6, and frozen MediaPipe Lite/Full SHA256 assets are preserved. No original ZIP is extracted, edited or uploaded.

A preflight of the original five anchor pairs found **8 invalid middle-index RGB↔Vicon references among 480 distinct sampled image-path instances** (A3 3, A4 5; each had two model variants). Replacing the wrong Vicon record with the unique source-valid record altered the corresponding *Vicon reference knee included angle* by mean 0.30° (A3 C1, 1), 0.92° (A3 C6, 2), 1.10° (A4 C1, 4), and 0.19° (A4 C6, 1), with maximum 1.72° across those instances. These are limited reference-difference checks, **not a full recomputation of earlier model MAE/correlations**; prior published exploratory estimates still need a separately controlled sensitivity audit before being reinterpreted.

## What is actually measured

[MediaPipe world landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) are reported in metres, with a midpoint-of-hips origin. The Vicon WCS numerical XYZ and model landmarks do **not** initially share a verified orientation, scale, origin, anatomical joint-centre definition or independent camera/Vicon exposure timestamps. Camera-space Vicon XYZ should not be treated as an additional independent physical ground truth.

The new metric is a **root-relative, train-aligned 3D six-joint geometric discrepancy**, not real-world absolute MPJPE:

1. Take left/right hip, knee and ankle 3D positions from **both** sources on correctly paired available images. Require all six MediaPipe joints visible at `>=0.5` and finite WCS reference coordinates. Apply model hip-centre subtraction and **separate ground-truth hip-centre subtraction to each frame**. This GT-dependent centering is an *evaluation-only oracle*, not deployable camera-only localization.
2. For each **participant × camera × model** combination, select 24 unique A3 D2 RGB JPEGs, then fit **one fixed proper rotation and positive scale** on the first 12 A3 sampled slots (at least six valid). Unlike conventional per-frame Procrustes, do **not** use test-frame joint coordinates to refit rotation/scale.
3. Freeze that one transform and evaluate the remaining 12 A3 slots as **within-action held-out frames** and all 24 A4 D2 slots as **cross-action held-out frames** for the same participant/camera/model. Require at least six valid test frames in each set; otherwise fail closed.
4. For each held-out frame, calculate Euclidean differences for six root-centred joints after transformation. Divide their mean by the **mean Vicon hip–knee segment length in that same frame**, then express as a **dimensionless percentage of femur length**. Also retain numerical discrepancies in raw WCS coordinate units *only in private local reports*. **Do not label the raw number millimetres without an independent validated unit conversion.**
5. Summarize **each participant–camera–model** first, then take equal-weight medians across the five historical calibration groups for exploratory comparison. Counts of images, model inferences and frames are **not independent calibrations**.

This transform learns *subject-specific A3 Vicon alignment* and therefore measures geometry transfer to A4 on the **same persons**, not independent subject-domain generalization. The method uses Vicon GT root positions on test frames for centering; it cannot report physical global joint location accuracy, calibrated camera/world XYZ, clinical joint kinematics, model-training data disjointness or synchronized timestamps. A lower normalized discrepancy is **not proof of clinically accurate 3D pose**.

## Run on authorized private research PC

Use a dedicated clean code checkout with the version-controlled `sumedipose.position_holdout` module, and original raw data under the approved research drive. Expected **960 model-image inferences**: five historical anchors × two camera views × two model variants × two actions × 24 unique images. Some sampled images may be excluded for missing detections/visibility. The 12 A3 training slots are fixed *before* seeing model predictions.

```powershell
$repo = (Get-Location).Path
$root = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path $repo "src"
python -m sumedipose.position_holdout `
  --root "$root\raw" --model-dir "$root\models" `
  --analysis-dir "$root\analysis" `
  --detailed-output "$root\analysis\A3_A4_rootrelative_holdout_private_20261117.json" `
  --aggregate-output "$root\analysis\A3_A4_rootrelative_holdout_aggregate_20261117.json"
```

The command freezes SHA256 against the existing A3 26-subject protocol; refuses destination files that already exist, and requires both results outside public Git under the designated analysis folder. The private detailed output contains only participant/view/model aggregate metrics and *no raw RGB, per-frame 3D coordinates, predicted landmarks, or fitted rotation matrices*. Do not upload either participant-level source or detailed JSON to GitHub, Drive, Second Brain, Notion or public websites. Any sharing of aggregate findings follows the established evidence review and dataset terms (see [DATA_POLICY.md](../DATA_POLICY.md)).

## Scientific acceptance gates

- [ ] All source JPEGs uniquely pair with a verified camera metadata/WCS index: `dup_num==factor`.
- [ ] All model weights match frozen hashes and sampled images are identical across Lite/Full variants.
- [ ] Alignment uses A3 **only**; holdout test frames and A4 are excluded from scale/rotation fitting.
- [ ] Five groups × two cameras × two models yield sufficient valid frames for **both** A3 holdout and A4 transfer.
- [ ] Results summarize *femur-normalized* root-relative discrepancy (no unverified mm, no MPJPE, no clinical accuracy).
- [ ] Preserve existing A3/A4 angular-MAE reports. Their 8 previously invalid RGB-to-coordinate references should be assessed separately with a frozen strict-pairing sensitivity analysis.
- [ ] Zero new independent calibration groups, no claim of validated absolute capture synchronization, anatomical equivalence or new patient/participant recording.

Any next attempt at true `MPJPE(mm)` requires independent proof of absolute WCS units, camera-to-Vicon position/orientation registration, exact frame timestamps, and a harmonized anatomical landmark reference — all unverified here.
