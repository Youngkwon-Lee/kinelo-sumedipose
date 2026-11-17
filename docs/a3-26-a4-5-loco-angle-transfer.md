# A3 26-participant → A4 five-anchor: held-out calibration-group conditional knee-angle test

**Status:** exploratory code and local-data validation only. No new real-world capture, no independent hardware-clock synchronization, no anatomically equivalent MediaPipe/Vicon reference.

## Question

The previous A3 first-half constant bias correction **often worsened held-out A4 deadlift**. An independent depth/marker-reference audit showed that rotations/reflections do not change a knee included angle, and anatomical landmark comparability is unresolved. Can a very small model that estimates residual 3D knee-angle bias from **the MediaPipe predicted angle alone** outperform the existing raw and constant-bias baselines on truly unseen **historical calibration groups** and a withheld action?

## Predeclared groups and methods

- Read official source `raw/calibration/external/subject_map.json`; exclude missing RGB S5/S10, leaving A3 26 participants in five existing groups (4,4,6,6,6). A4 tests use the existing one anchor participant per group.
- C1 frontal and C6 rear views, MediaPipe Lite/Full, A3 squat D2 and A4 deadlift D2, 24 strict `dup_num == factor` selected RGB JPEGs in each recording. **104 A3 + 20 A4 = 124 recording-model runs, 2,976 attempted inferences.** All evaluations require six visible hip/knee/ankle landmarks in each used frame.
- **Five leave-one-calibration-group-out folds:** hold out **all** A3 participants of a target historical group *and* its A4 anchor. Train on the **other four groups' A3 D2 only**, never on any target label/A4 label. Each training group and participant is equally weighted (frames are never treated as independent calibration sessions).
- Compare exactly three frozen methods, no test-set search:
  1. **Raw** MediaPipe world 3D included knee angle.
  2. **Constant** correction: group-equal signed angle residual mean `e = model_angle - Vicon_WCS_angle`.
  3. **Linear ridge** correction: `e_hat = intercept + slope * ((model_angle-150°)/40°)`; intercept unpenalized, slope ridge penalty fixed at **0.50** before evaluating test groups. Corrected prediction = raw model angle minus `e_hat`.
- Prediction uses **only model-predicted knee angle**. Vicon test labels are used solely to calculate held-out MAE and descriptive bands by reference angle (under 120°, 120–150°, at least 150°). Report extrapolation outside training-model-angle range and adjusted predictions outside 0–180°; **never clip** and never silently impute.
- A3 heldout groups test **new participants with same action**. A4 tests **those same previously unseen groups' one historical anchor in another action**. These are *not* newly acquired external calibration days; the five groups have been analyzed in prior exploratory research. The learned coefficients cannot be transferred to Kinelo or clinical use.

## Strict source and identity gates

Reuse `sumedipose.position_holdout.read_pose_frames` to enforce source-valid `dup_num == factor`, the original 24 distinct JPEG files and frozen model asset SHA256. Check every one of the **124 previous private strict-source-pairing evaluation reports** for sample/model/hash identity, image attempts, pose detections and erroneous midpoint-reference counts. Abort on mismatches or incomplete five-fold cohort. No updating previous raw/source/benchmark reports.

### Windows local-only command

```powershell
$base = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m sumedipose.loco_angle_transfer --root "$base\raw" --model-dir "$base\models" --analysis-dir "$base\analysis" --private-output "$base\analysis\A3_26_A4_5_LOCO_angle_private_20261118.json" --aggregate-output "$base\analysis\A3_26_A4_5_LOCO_angle_aggregate_20261118.json"
```

Both output files must not exist already; the command will refuse to overwrite. The private report contains participant/run summary identity, fold fits and group metrics, **not** images, raw XYZ, model predictions per frame or raw motion time series. The aggregate has no participant identifier. All data-derived files remain only on the authorized Windows D: research drive. GitHub and Second Brain may receive code/tests and **reviewed de-identified five-group summary results**, never source/raw or per-person JSON.

## Interpretation and action

A positive *within-historical-group* result cannot validate real hardware camera-to-Vicon timing, identical anatomical reference landmarks, generalization to new calibration dates/device, physical MPJPE in mm, patient-level accuracy or a clinically appropriate cutoff. If linear ridge does not consistently improve held-out A3 and A4 groups over raw and constant, **do not promote automatic angular bias correction**.

Anatomical harmonization is a separate research gate requiring a validated source definition linking MediaPipe model landmarks and Vicon joint centres; substituting Vicon skin markers for source joint centres did not establish that equivalence.
## Windows real-data execution and independent readback — 2026-11-18

**Status:** The fixed analysis completed successfully, with **124 model-recording evaluations**, **2,976 attempted model-image inferences**, **2,949 pose detections** and **2,881 valid six-visible-joint frames**. All 124 prior source-valid recording identity/model-SHA/JPEG-reference and detection checks passed. Five calibration groups remain historical (zero new independently collected calibration days).

The program uses the original **26 A3 participants** and the **five A4 anchors**. In each fold, *all subjects* sharing the held-out calibration group are excluded from training, rather than excluding only one participant. Training weights are equal by source calibration group, then participant, then observation. No A4 Vicon reference labels or held-out A3 reference labels enter either calibration fit.

### Leave-one-calibration-group-out group-equal MAE

Five group-mean MAEs equally weighted, degrees. The constant and linear values are corrected test errors; the raw column is without correction.

| Camera/model | A3 raw | A3 constant | A3 linear ridge | A4 raw | A4 constant | A4 linear ridge |
|---|---:|---:|---:|---:|---:|---:|
| C1 Lite | 13.2819 | 13.1570 | **11.8495** | 13.9013 | 13.0780 | **12.5609** |
| C1 Full | 14.6153 | 14.3348 | **12.3662** | 15.6922 | 14.4140 | **13.6404** |
| C6 Lite | 25.2050 | 24.5864 | **22.9164** | **17.4423** | 22.9269 | 21.9838 |
| C6 Full | 22.2853 | 21.0515 | **20.4617** | **13.7485** | 18.9025 | 18.2951 |

- **Unseen-group A3:** The angle-dependent ridge model reduced the five-group mean MAE in all four view/model conditions; C1 Lite, C1 Full, C6 Lite each improved in **5/5 groups**, C6 Full **4/5**. This is *within the same squat action*, not new acquisition-day validation.
- **Unseen-group A4:** C1 Lite and Full ridge showed mean improvements of **1.3404°** and **2.0519°**, respectively, each **4/5 groups improved**. C6 Lite and Full became **4.5416°** and **4.5466° worse** on average, each **1/5 groups improved**. The other four groups worsened.
- A4 test predictions never fell outside the full model-angle ranges observed in their four-group A3 training sets (zero measured range extrapolations). This does *not* prove identical motion-phase or viewpoint distributions across actions.
- The prediction is a 3D **included-angle scalar**; uniform global rotation, translation, reflection, or positive scaling of a model's three joint points cannot change it. The conditional correction does not establish matching anatomical landmarks or fix 3D point positions.

### Stratification exposes opposing angle-dependent error effects

The reference Vicon 3D included-angle bands below are **post hoc scoring groups only**; Vicon band labels were not input to either correction model.

- **C6 rear / Lite, A4:** In the deeply flexed `<120°` band (data from **4 historical groups**, 32 valid knee-angle observations), ridge MAE **54.1451° → 35.2522°**; in the nearly extended `≥150°` band (5 groups, 136 observations), **6.4965° → 20.7500°**. The conditional correction helps deep flexion but severely harms near extension.
- **C6 rear / Full, A4:** `<120°` (4 groups, 28 angles) **40.5789° → 21.7872°**; `≥150°` (5 groups, 136 angles) **5.8506° → 19.0220°**. The same reversal occurs.
- **C1 front / Lite, A4:** `<120°` (4 groups, 23 angles) **19.2679° → 22.4341°**, while `≥150°` (5 groups, 137 angles) **13.4043° → 10.3960°**. Thus even apparently improved overall C1 error is not uniformly improved across movement phases.

**Scientific decision:** A shared view-independent angular correction is **not validated**. Overall performance gains at a view/model level can hide major deterioration in a clinically relevant posture subgroup. Do not deploy the learned ridge coefficients, set medical thresholds or declare causal biomechanical error attribution. The fixed `λ=0.50` was predeclared, not optimized on held-out test groups; the small historical cohort, prior exploratory reuse and differences in anatomical landmark conventions still limit inference.

### Reproduction and privacy receipt

- Windows Python synthetic/legacy tests: **124/124 passed**.
- Independent local readback from saved group-level private report, without re-running model inference: **480/480 aggregate and stratified checks passed**, maximum absolute mismatch due to rounding **0.0000494°**.
- Authorized local output pointers, under `D:\Research\PublicDatasets\SUMediPose\analysis`: `A3_26_A4_5_LOCO_angle_private_20261118.json`, `A3_26_A4_5_LOCO_angle_aggregate_20261118.json`. Original RGB/Vicon/MediaPipe landmarks and per-frame predictions are not saved or uploaded.
- Remaining independent blockers: synchronized physical exposure timestamps, a validated anatomical equivalent joint-label protocol, unseen calibration dates/devices and model training-data provenance. Only nonidentifying group aggregate interpretations may enter the existing Second Brain Candidate.
