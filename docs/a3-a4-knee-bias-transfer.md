# A3-trained knee angle bias → A3/A4 held-out transfer test

**Question:** An earlier 26-person A3 D2 study found systematic 3D knee included-angle errors, especially at deep flexion. Is a single **supervised Vicon-reference angular offset learned exclusively on A3 training samples** transferable to held-out squat A3 and a distinct deadlift action A4?

This is **not** anatomical landmark harmonization, a model training procedure, a clinically deployable correction, or independent medical-device accuracy validation.

## Physical/mathematical limitation

The 3D included knee angle is **invariant under any orthogonal rotation or reflected axis convention and positive uniform scale**. Rotating or mirroring MediaPipe's entire XYZ joint triplet cannot, by itself, lower its 3D included-knee-angle error. The prior code's different WCS/CCS axis handedness and model parity holdout can matter for interpreting *joint positions*, but they do not establish the cause of the 3D included-angle bias. Neither Vicon model joint centres nor MediaPipe anatomical landmarks were independently redefined/matched by this experiment.

## Fixed sampling and training/test boundaries

- Five **historical** session anchor participants (S1, S6, S11, S17, S23), C1/C6, Lite/Full, A3 D2 and A4 D2. No new independent session. 20 participant/view/model combinations × 2 actions × 24 unique RGB samples = **960 attempted inferences**.
- Reuse `sumedipose.position_holdout.read_pose_frames()` with **strict source-valid `dup_num==factor` JPEG↔Vicon record associations**, previously verified model assets and only frames where all six lower-limb MediaPipe landmarks satisfy >=0.5 visibility.
- **TRAIN:** A3 first 12 selected sample slots only. **TEST:** remaining A3 last 12 and all A4 24, with held-out reference angles never used in calibration fitting.
- Fit exactly one scalar offset (mean predicted 3D included angle minus Vicon 3D included angle) to all valid left/right knee angles from the A3 training half. Correct prediction by subtracting that offset. **Do not clamp** values to 0–180°, and report how many corrected values leave the physiological angle range.
- Compare (A) own-participant A3 fitted offset and (B) a *leave-one-calibration-group-out* offset, equal mean of the four **other historical groups' A3 first-half offsets**, never fitting the held-out group. This is limited heldout historical-group evaluation, not independently acquired external subject/device generalization.
- Compare paired baseline and corrected MAE and number of groups improved/worsened for A3 test vs A4 transfer, C1/C6 and Lite/Full. Also report exploratory Vicon 3D included angle bins <120°, 120–150°, >=150° with valid group coverage; **not clinical thresholds**.
- Fail closed unless the previously saved original local `A3_A4_rootrelative_holdout_private_20261117.json` 20 rows reproduce the strict source-valid image selection, attempted/detected poses, and valid six-joint A3/A4 subset counts. Unlike the earlier 26-person summary, this experiment keeps the **six-visible-joint subset fixed** to support paired transfer comparisons.

## Authorized Windows-only experiment

Run from a clean scoped `kinelo-sumedipose` checkout, never overwriting existing results:

```powershell
$root = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m sumedipose.knee_bias_transfer --root "$root\raw" --model-dir "$root\models" --analysis-dir "$root\analysis" --private-output "$root\analysis\A3_A4_knee_bias_transfer_private_20261117.json" --aggregate-output "$root\analysis\A3_A4_knee_bias_transfer_aggregate_20261117.json"
```

The private report stores only participant/view/model **summaries** and a code SHA256 receipt; **no RGB, source 3D coordinates, landmark XYZ, fitted spatial matrices or per-frame predictions**. Both files remain under the local Windows research D: analysis tree. No participant-level files belong in GitHub, Drive, Notion or Second Brain. Report only de-identified group aggregates outside that storage.

A success in transfer would show that **Vicon-labeled offset correction** reduces error on withheld motion. It would not show whether MediaPipe got anatomical joint centres or physical camera depth right. If a constant offset helps A3 but fails A4 or new groups, it is insufficient as a general correction; no product/clinical threshold should be set from five historical anchor groups.

## Vicon joint definition proxy sensitivity (source-only)

An additional **read-only** experiment computes knee angles from the *same*
SUMediPose Vicon WCS records using different source marker definitions.

Original reference = `HJC-KJC-AJC` (derived Vicon hip/knee/ankle joint centres).
Alternative references keep the **same HJC** but replace `KJC` with the
source `KNE` surface marker, `AJC` with the `ANK` surface marker, or
both together. Repeat for left/right, with source-valid `dup_num==factor`
records and the exact same A3 26-person / A4 five-anchor **24 unique
image name selection** as the prior studies.

**IMPORTANT:** KNE and ANK are *skin markers*, NOT anatomically
equivalent to the model's predicted knee/ankle landmarks; the alternative
angles are **not clinical ground truth**. The absolute difference merely
quantifies a geometry sensitivity to the chosen definition in Vicon.
It does not estimate a MediaPipe joint-position error, prove that landmark
mismatch causes previous signed angle bias, nor correct 3D model output.

Run on Windows from the authorized research checkout:

```powershell
$root = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m sumedipose.marker_reference_sensitivity --root "$root\raw" --analysis-dir "$root\analysis" --private-output "$root\analysis\A3_A4_marker_reference_sensitivity_private_20261117.json" --aggregate-output "$root\analysis\A3_A4_marker_reference_sensitivity_aggregate_20261117.json"
```

No MediaPipe inference or new source data download. Detailed
participant-level geometric summaries remain **Windows-local**; release only
five-group aggregate values in the pre-existing research Candidate.

## Executed Windows-local results — 2026-11-17

### Constant angular offset transfer

The existing strict source-verified A3/A4 root-relative six-joint study was replayed with **40 model-action recordings, 960 attempted RGB inferences and 933 detected poses**. Each one of the **20 anchor×view×model** combinations reproduced its prior source-selected image/pose-detection/six-joint-valid count. A separately run readback recalculated **80/80 stored aggregate and count checks** across the 16 cell summaries, with maximum rounding difference **0.000049°**.

**A4 holdout: median per-session MAE before and after a *same-subject A3 first-12* bias correction.** All A4 source Vicon angle labels were withheld from fitting.

| Camera | Model | Baseline A4 MAE | Corrected A4 MAE | Groups improved/worsened |
|---|---|---:|---:|---:|
| C1 | Lite | 13.3058° | 17.7745° | 1 / 4 |
| C1 | Full | 15.1197° | 19.1995° | 1 / 4 |
| C6 | Lite | 15.4066° | 23.8394° | 1 / 4 |
| C6 | Full | 14.8857° | 19.8204° | 1 / 4 |

**A4 cross-calibration-group test:** when the scalar was fitted on the **four other** historical calibration anchors' A3 first halves, median A4 corrected MAE was C1 Lite **17.5610°**, C1 Full **18.8016°**, C6 Lite **25.5396°**, C6 Full **23.1829°**. Group-level paired improvements: C1 Lite/Full **0/5**, C6 Lite/Full **1/5**. Both approaches frequently made A4 worse; no deployable angle offset is validated.

The A3 withheld-half test did not consistently improve either: **C1 Lite subject-specific** median MAE moved **14.9352° → 14.3496°**, but **C6 Lite** moved **25.5954° → 27.3721°**. Avoid treating a constant A3 offset as a stable correction across motion phase, camera, participant or action.

### Source Vicon anatomical-reference proxy sensitivity

Using source WCS **without new AI inference**, strict source-valid 24-JPEG-selected indices covered **52 A3 recording views for 26 participants** plus **10 A4 views for five anchors**: **62 recording views / 1,488 JPEG instances / 2,976 left-right reference knee angles**. The prior midpoint-index reference mistakes count was **21/1,488**. All **128 independent group-reaggregation/count checks** matched the initial aggregate (max rounding difference 0.00005°).

**Change in the Vicon included knee angle when joint-centre landmarks are replaced by source skin markers — mean of five group-mean absolute differences (degrees):**

| Action / view | Knee centre → KNE | Ankle centre → ANK | Both → surface markers |
|---|---:|---:|---:|
| A3 C1 | 7.5848° | 2.2588° | 4.5312° |
| A3 C6 | 7.5640° | 2.2480° | 4.4735° |
| A4 C1 | 7.4255° | 2.3939° | 4.2400° |
| A4 C6 | 7.4545° | 2.2878° | 4.2545° |

These skin markers are **not** validated anatomical homologues of MediaPipe landmarks. The combined change can be smaller than a knee-only change because geometric effects cancel. These numbers must **never** be added to the model MAE to claim a proportion explained. This is reference-definition *sensitivity*, not anatomical source-of-error attribution.

The coordinate-axis rotation/reflection result is invariant for an included knee angle; a coordinate **handedness** correction cannot by itself fix that scalar angle. Deeper flexion and differences in model/Vicon landmark position and motion dynamics are research hypotheses. Physical camera–Vicon exposure timing, independent subject/device validation and a clinically validated reference remain outstanding.

### Private evidence pointers

Keep all under `D:\Research\PublicDatasets\SUMediPose\analysis` on the approved Windows research PC. **Do not upload their contents**:

- `A3_A4_knee_bias_transfer_private_20261117.json` (per-anchor summary), `A3_A4_knee_bias_transfer_aggregate_20261117.json` (de-identified group summary)
- `A3_A4_marker_reference_sensitivity_private_20261117.json` (per-participant summary), `A3_A4_marker_reference_sensitivity_aggregate_20261117.json` (de-identified group summary)

Code-only tests and source methodology can be merged. These experiments neither train a new AI pose model nor establish physical-world MPJPE(mm), clinical 3D accuracy, model-data independence or actual camera/Vicon clock synchronization.
