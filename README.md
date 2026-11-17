# Kinelo × SUMediPose

Open-source **code-only** toolkit for exploratory analysis of the [SUMediPose](https://doi.org/10.1016/j.dib.2025.111579) dataset, with an initial focus on **A3 (squat)** RGB images, world-coordinate (WCS) 3D markers, and camera-coordinate 2D/3D markers.

> **Research status:** Early prototype. Not a medical device, clinical decision tool, or validated motion-assessment system. No participant images, motion data, or original dataset archives are included in this repository.

## What this does

- Read existing SUMediPose ZIP archives in-place; **never extract or modify raw data**.
- Identify A3 squat samples by camera, subject, and repeat (for example, `C1/S1/A3/D1`).
- Summarize records, frame identifiers, and marker IDs.
- Calculate preliminary hip–knee–ankle included angles from WCS 3D coordinates, in degrees.
- Optionally save a **local** three-panel PNG showing one RGB frame, camera marker overlays, and exploratory WCS/camera 3D marker clouds.
- Run dependency-light tests without downloading the dataset.

## Local setup

Requires Python 3.10+:

```powershell
git clone https://github.com/Youngkwon-Lee/kinelo-sumedipose.git
cd kinelo-sumedipose
python -m pip install -e ".[test,viz,image]"
python -m pytest -q
```

Data must be acquired separately from the official SUMediPose source, following its own terms. This repo assumes a dataset tree like:

```text
D:\Research\PublicDatasets\SUMediPose\raw\
  frames\C1\S1\C1S1A3D1.zip
  WCS\S1.zip
  internal_data\C1\S1.zip
```

Inspect and calculate example knee angles:

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw"
```

Generate a **local, non-publishable** preview (the image contains participant imagery):

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --preview "D:\Research\PublicDatasets\SUMediPose\analysis\C1S1A3D1_preview.png"
```

The image is intentionally created **outside the Git repository**. Do not publish it without assessing dataset terms, informed consent and ethics requirements.

## Important interpretation caveats

1. WCS and camera record lists can both contain 1,205 items while the RGB sample has 928 files; `frame_num` values are **not necessarily identical**. Index-aligned WCS/camera records in the prototype are only an exploratory preview, **not validated temporal synchronization**.
2. WCS and camera 3D coordinates appear to use different scales. **Do not assume identical length units** or use absolute distances across systems without checking the data documentation and calibration.
3. Marker IDs such as `LHJC`, `LKJC`, `LAJC` and their right-side counterparts are used for example included knee angles, **not clinical knee flexion** or independently validated biomechanics.
4. Annotations and calibration can differ across subjects and recordings. Benchmarking requires frame alignment, projection validation, coordinate conventions, quality control, and independent evaluation.

## Roadmap

- [x] Code-only repository and strict data exclusion.
- [x] Original ZIP sample reader, preliminary geometry and preview.
- [ ] Explicit validated frame synchronization and dropped-frame handling.
- [ ] Documented calibration and coordinate conversions.
- [ ] Verified joint kinematics with anatomical conventions.
- [ ] Benchmark adapters for MeTRAbs / GVHMR / other pose models.
- [ ] Per-subject evaluation, uncertainty and bias reporting (subject-wise splits).

## Privacy and dataset

See [DATA_POLICY.md](DATA_POLICY.md). Research originals must remain on local storage, and neither the archives, per-subject metadata nor preview images belong in Git history.

## License

Our **original code and documentation only**: MIT; see [LICENSE](LICENSE). This does **not** grant rights over SUMediPose data or third-party software.

## A3 RGB / camera / WCS alignment audit

The local audit reads original archives **in place** without extracting or rewriting them. Run tests with `python -m pytest` after installation. For one camera/subject/repeat:

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --camera C1 --subject S1 --repeat D1 --audit
```

To save derived, **local-only** reports (do not upload these subject-level outputs to Git):

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --camera C1 --subject S1 --repeat D1 --audit --audit-json "D:\Research\PublicDatasets\SUMediPose\analysis\C1S1A3D1_alignment_audit.json" --mapping-csv "D:\Research\PublicDatasets\SUMediPose\analysis\C1S1A3D1_alignment_map.csv"
```

The CSV maps each WCS **list index** to a camera metadata record, `frame_num_w_drop`, camera `frame_num`, and referenced RGB ZIP member. An image member can be **reused** for multiple coordinate records; do not count those as independent photographs or assume one RGB image for every WCS record. Missing referenced RGB members are explicitly marked, **not interpolated or downloaded**.

The audit exposes three distinct results:

- `record_index_alignment_supported`: WCS and camera 3D records have matching marker IDs and near-identical included knee angles at the **same list index** (a heuristic check; their 3D coordinates may derive from the same source).
- `rgb_mapping_complete`: all camera metadata image references exist in the RGB ZIP with valid expected names.
- `complete_rgb_alignment_audit_pass`: both checks and sequence diagnostics pass; this is **structural** validation, not independently validated video/Vicon timestamp synchronization.

`physical_timestamp_synchronization_verified` remains **false** until exposure timestamps, calibration and hardware clock or independently measurable temporal events have been checked. The first/last RGB frame numbers can differ substantially from WCS frame numbers; never join by equality of their numeric `frame_num` fields. See `DATA_POLICY.md` before sharing any outputs.

## RGB-motion timing plausibility and calibration inventory

The optional image workflow uses **real RGB JPEG pixels** and forward/backward Lucas-Kanade optical flow. It compares upper-torso image feature motion against the source-provided projected C7/CLAV displacement. Each pair spans six **unique available RGB images** by default, not six WCS frames or known milliseconds.

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S1 --camera C1 --repeat D1 --motion-audit --stride 6 --motion-json "D:\Research\PublicDatasets\SUMediPose\analysis\C1S1A3D1_motion_audit.json"
```

Read the original external calibration metadata and matrix structures:

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S1 --calibration-inventory --calibration-json "D:\Research\PublicDatasets\SUMediPose\analysis\S1_external_calibration_inventory.json"
```

- Aggregate reports are **local-only** JSON. No RGB images or per-frame pixel tracks are saved or published.
- Missing RGB references, including known terminal frames, are **masked** rather than interpolated, generated or re-downloaded. The same image can map to multiple 3D records.
- Lag units are sampled image **pairs**, not frames or seconds. The exploratory screen requires >=15 valid pairs, lag-0 Pearson correlation >=0.8 and best across tested lags, plus median absolute displacement error <3 pixels at the resized analysis resolution. These are **heuristics**, not clinical thresholds.
- The tracking ROI is selected with source-provided, WCS-derived camera 2D markers. Image features move independently in pixel space, but torso localization is **not blind**. Positive evidence supports motion correspondence plausibility only.
- A failure or nonzero best lag may reflect view occlusion, feature tracking limitations, image sampling, calibration, or real temporal mismatch. It is **not** proof of a hardware clock offset.
- Calibration inventory verifies source CSV entries and the existence, finite values and shapes of R/T/intrinsics matrices. Source calibration error values have **unverified units**; independent RGB reprojection accuracy has not been tested.
- Physical exposure timestamp sync, 2D/3D model benchmark accuracy and clinical validity remain **unverified** regardless of the exploratory score.

Install the optional OpenCV support using `pip install -e ".[image]"`; no dataset download occurs.

## View-dependent ROI diagnostics and timestamp evidence

The optical-flow test is sensitive to which pixels are allowed to contribute tracking points. A rear view can have different feature texture and occlusions than the front view. **Do not choose the best-performing ROI after inspecting a test set and call it independently validated timing**. Preserve the original setting, compare controlled alternative ROI sizes at a fixed image width, and inspect a separate recording before treating the change as a stable observation.

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --camera C6 --subject S1 --repeat D1 --motion-audit --image-width 288 --roi-markers C7 CLAV --roi-half-width 16 --roi-half-height 18 --motion-json "D:\Research\PublicDatasets\SUMediPose\analysis\C6S1A3D1_tight_roi.json"
```

To separate the **effective ROI size** from the resized image resolution, scale the ROI dimensions proportionally when doubling width (for example 288px width with 32×36px half-ROI vs 576px width with 64×72px half-ROI). Lag output units remain sampled image pairs, **not** camera frames or milliseconds. Compare front and rear cameras, as well as repeated recordings, before making a causal claim.

Compare the dataset's **manually annotated calibration-wand LED positions** to its own provided `wand_xy` projections:

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S1 --repeat D1 --wand-fit --wand-json "D:\Research\PublicDatasets\SUMediPose\analysis\S1_wand_annotation_fit_residuals.json"
```

The five-point wand residuals may use points originally involved in **fitting** camera projection parameters. Their RMS pixel error is an **in-sample reference-fit diagnostic**, **not** an independently held-out calibration/reprojection accuracy result. The physical wand reference (A0) images may not be present in the A3-only download; absent source images must not be fabricated.

Check whether camera and WCS records, the main CSV header, and up to three available JPEG EXIF records expose real capture times:

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --camera C6 --subject S1 --repeat D1 --timestamp-evidence --timestamp-json "D:\Research\PublicDatasets\SUMediPose\analysis\C6S1A3D1_timestamp_evidence.json"
```

This bounded evidence inventory is **not an exhaustive search for all possible source log files**. Camera `frame_num`, `frame_num_w_drop`, ZIP modification dates, and post-hoc stomp alignment do **not** establish a common hardware clock. All results retain independent synchronization as unverified until original frame-timestamp logs or hardware sync-event evidence is available. Only aggregate JSON leaves the original ZIP reader; keep all per-record outputs local.

## Reproducible pose-model feasibility benchmark

Optional local-only `MediaPipe Pose Landmarker` benchmark: two Google-released weights (`Lite` and `Full`) each output **both 2D image landmarks and 3D world landmarks from the same model**. The two weight variants are distinct model sizes, not independent 2D and 3D architectures.

Install optional dependencies: `python -m pip install -e ".[benchmark]"`. Download the official model `.task` asset separately (for example, from [Google's Pose Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)), place it **outside this Git repository**, and verify its integrity.

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --camera C1 --subject S6 --repeat D2 --pose-baseline --model-task "D:\Research\PublicDatasets\SUMediPose\models\pose_landmarker_lite.task" --max-frames 24 --benchmark-json "D:\Research\PublicDatasets\SUMediPose\analysis\S1A3_pose_S6_C1_lite.json"
```

Each run selects up to 24 **evenly spaced unique available RGB JPEG files**. Their camera metadata supplies a coordinate-record index; the WCS index is checked for equal record count. Only valid pose detections with all hip/knee/ankle landmarks visible at `>=0.5` are scored. Results contain only **aggregate** joint-angle absolute error, RMSE, median, p90, and missed-detection counts; no source images or per-frame landmarks are saved.

- **2D metric:** the 2D included hip–knee–ankle angle estimated from model image landmarks is compared with the 2D angle formed by Vicon joint-centre projections (`LHJC/LKJC/LAJC`, equivalent on right), in pixel coordinates (aspect ratio preserved).
- **3D metric:** the 3D included angle estimated from model hip-centred world landmarks is compared with the angle formed by Vicon WCS 3D joint centres. Angle is invariant to overall scale and rigid rotation.
- **Do not directly compare 2D and 3D MAE as model superiority:** the geometric reference angle is different in 2D vs 3D, and MediaPipe's anatomical landmarks do not match the Vicon joint-centre definitions exactly. Within one reference modality, compare Lite and Full using identical images, labels and visibility rules.
- **No clinical validation:** no held-out model-training provenance audit, calibration into Vicon coordinates, independent capture timestamp proof, population-level statistical inference, or clinical knee ROM definitions were established. This is an offline feasibility pilot, not a medical device assessment.

Record the exact model URL, SHA256, MediaPipe runtime version, source data selection and per-participant grouping in a **local-only** protocol. Do not publish identifiable RGB videos, per-participant predictions, private manifests or evaluation images.

## Independent-of-fit stereo checkerboard geometry (not Vicon time synchronization)

The SUMediPose [Harvard Dataverse dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GRRROM) provides separate **external calibration frame archives** (e.g. `calibration/external/frames/S1.zip` and `S6.zip`) that are **not** part of the existing squat A3 ZIP set. Acquire archives separately under the source CC BY-NC-SA 4.0 terms. Never commit the originals, extracted JPGs, or per-participant evaluation records.

One calibration ZIP holds eight groups of 60 JPEGs (480 total), for the four original stereo pairs C1/C2, C1/C3, C6/C4 and C6/C5. `raw/calibration/external_calibration.csv` lists the 5–6 frames used to fit each pair's original R/T; the remaining source frames are withheld for evaluation. The original calibration JPGs need a **90° counterclockwise orientation transform** before comparing detected board points with the source's portrait camera intrinsics.

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S1 --stereo-holdout --paired-camera C5 --external-frames-zip "D:\Research\PublicDatasets\SUMediPose\source_calibration\external_frames\S1.zip" --stereo-json "D:\Research\PublicDatasets\SUMediPose\analysis\S1_stereo_C5_holdout.json"
```

The audit detects the 8×6 internal chessboard intersections from images *in memory*. Symmetry/orientation choice for point ordering is fixed **using source fitting frames only**. It then computes the median of the symmetric **Sampson epipolar distance** (pixels in the undistorted camera plane) over each separately held-out image pair, using the saved source R/T and intrinsics **without refitting them**. The output includes holdout sample detection coverage, median and p90 frame-level residual, and an explicit missing/undetected count.

Epipolar geometry consistency is **not 3D point reprojection accuracy**: estimated board plane pose, stereo triangulation, and independent Vicon joint-centre error are not established. Same-numbered calibration images may not have independently proven exposure simultaneity. The audit is sensitive to chessboard detection failures, board symmetries and source frame selection; report coverage, fit residual and holdout residual together. Do not infer camera/Vicon hardware clock synchronization or clinical validity from a low epipolar residual.

Install with `python -m pip install -e ".[image]"` to provide OpenCV. No dataset download is triggered by the CLI.

## File-index lag sensitivity and triangulated checkerboard consistency

**Source correction (2026-11-16):** The [SUMediPose peer-reviewed paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088771/) (Camera System Calibration, section 5.3.2) explicitly reports an **A2 calibration board with 60 mm squares**. The [authors' calibration pipeline](https://github.com/chris-schreuder/SUMediPose_pipeline/blob/main/code/calibration/extrinsic.py) uses an 8x6 interior-corner board with world_scaling = 1 per square. Thus **one original calibration-square unit corresponds nominally to 60 mm** for this board. This corrects older statements that the square length is unavailable. The board was **not independently remeasured here**, nor does nominal conversion establish actual Vicon-world 3D point accuracy. Original source R/T/K/distortion remain frozen.

### Run on a previously verified, separate official external-calibration ZIP

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S6 --stereo-lag --paired-camera C5 --max-calib-lag 5 --external-frames-zip "D:\Research\PublicDatasets\SUMediPose\source_calibration\external_frames\S6.zip" --stereo-lag-json "D:\Research\PublicDatasets\SUMediPose\analysis\S6_G65_lag_board3d_cli.json"
```

- The **-5 to +5 offsets** are *file/image-number* shifts in one source calibration group, **not milliseconds or proof of camera-exposure clock alignment**. Source fit frames are excluded from **both** cameras at all tested offsets.
- Source fitting frames alone choose checkerboard orientation; all holdout frames retain that choice. The scan reports Sampson epipolar error median and 90th percentile at each shift, detection coverage, and a fairer **matched-frame-anchor** comparison against zero shift because available images change as the offset changes.
- For zero index shift, detect an 8×6 checkerboard and independently triangulate its corners from original stereo intrinsics/extrinsics (no refitting). Reprojection back into the **same two images** gives a **stereo self-consistency** residual in pixels. The reconstructed checkerboard's neighboring edge lengths and planarity are measured in arbitrary **source chessboard-square units**.
- A low two-view reprojection residual is **not independent 3D physical truth**. Source board-square units can be converted into **nominal mm** using the authors' published 60-mm square length, but this is **not an independent 3D accuracy estimate**. Direct remeasurement of the physical board and common Vicon/camera capture-clock evidence are unavailable. Differences may reflect board detection, lens distortion, calibration stability and capture mismatch.
- The source and any derived subject-level geometry files stay outside this public Git history. The code returns *aggregate scalars only*, without images, detected corners or reconstructed per-frame point clouds. No model training, new image download or clinical analysis is triggered.

## Camera-pair lens vs relative-extrinsic uncertainty audit

This local-only diagnostic is intended for a camera pair with **large held-out epipolar residuals**, e.g. rear C6–C5. It separates two alternative explanations: inaccurate *individual-camera* checkerboard detection/pose fitting versus inconsistent *between-camera* relative rotation/translation. The measurements are diagnostic, not causal proof.

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S6 --paired-camera C5 --calibration-uncertainty --external-frames-zip "D:\Research\PublicDatasets\SUMediPose\source_calibration\external_frames\S6.zip" --calibration-uncertainty-json "D:\Research\PublicDatasets\SUMediPose\analysis\S6_G65_calibration_uncertainty.json"
```

The audit uses only the **non-fit 8×6 checkerboard JPEG pairs** and freezes source stereo `R/T`, intrinsics and lens distortion coefficients. It applies source JPEG rotation and the checkerboard corner symmetry selected using fitting frames. For each pair of held-out images it reports:

- Sampson epipolar residual with lens coefficients applied versus without rectifying lens distortion; the change quantifies the distortion correction's effect rather than assuming it fixes the camera relationship.
- The pixel shift due to lens correction in **each camera**, and the checkerboard PnP reprojection residual for **each camera fitted separately to its own detected corners**. PnP reprojection here is *in-image fit*, not independent accuracy of the intrinsics.
- How far the **source's frozen inter-camera R/T** projects the checkerboard seen in camera A away from the corresponding observations in camera B (cross-camera pixel error). This can respond to calibration drift, board motion between exposures, detector error and image synchronization.
- Relative rotation angle difference (degrees) and translation difference (source board-square units) between the two independently PnP-fitted board poses and the original stereo transform. These are **not independent physical tracking or Vicon-world truth**.

No cross-camera timing/clock offset, rolling-shutter model, absolute Vicon joint error or clinical accuracy is inferred. The paper **does** report a nominal 60-mm board square and the source calibration uses world_scaling = 1, so stereo translations can nominally be converted to mm. Direct physical board metrology and independent absolute spatial accuracy remain **unverified**.

Only aggregate summary statistics are returned. Never publish JPGs, camera-specific point traces, private study outputs or source ZIPs. A high cross-camera error alongside a low per-camera PnP fit **prioritizes relative stereo calibration or capture-conditions investigation**, but cannot by itself determine which is responsible.

## Third calibration-session check and corrected nominal board scale

The prior S1/S6 sessions do not establish calibration quality across all acquisition days. Evaluate the **S11** calibration-day tag using the **same** frozen source R/T/K/distortion and fit-frame exclusions:

    sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --subject S11 --paired-camera C5 --calibration-uncertainty --external-frames-zip "D:\Research\PublicDatasets\SUMediPose\source_calibration\external_frames\S11.zip" --calibration-uncertainty-json "D:\Research\PublicDatasets\SUMediPose\analysis\S11_G65_calibration_uncertainty.json"

The original dataset and per-session results remain local. The **paper-reported 60-mm square length** corrects the prior assertion that the board scale is unknown; it does **not** independently validate camera-to-Vicon 3D registration, millimetre reconstruction accuracy or exact exposure-time synchronization. The board_scale module exposes the nominal conversion and the separate unverified ground-truth status.


## Calibration-stratified pose benchmark (exploratory)

Use the **five-session** calibration summary (S1, S6, S11, S17, S23) to explore whether C6-C5 stereo epipolar residuals covary with knee-angle MAE from **monocular** MediaPipe Lite/Full pose inference. This is not a causal test or clinical quality gate: stereo calibration and monocular model error are distinct quantities, and every session is associated with its own subject/recording.

**Input files stay in the private Windows analysis folder** (never check in the results). The comparison reads:

- S1_S6_S11_S17_S23_calibration_five_session_overview.json
- S1A3_mediapipe_lite_full_protocol.json (model asset SHA256, D2 and 24-image sampling)
- S1A3_pose_SUBJECT_CAMERA_VARIANT.json for each S1/S6/S11/S17/S23, C1/C6, lite/full (20 model reports total)

The original 192-image pilot only evaluated **S6/S11/S17/S23**. S1 was previously used to tune optical-flow ROI settings and was not part of that heldout pilot. Do not claim a five-session association until S1 has been evaluated with the same model weights and sampling. If missing, create only the four S1 aggregate results on the authorized Windows computer:

~~~powershell
$root = "D:\Research\PublicDatasets\SUMediPose"
foreach ($camera in @("C1", "C6")) {
    foreach ($model in @("lite", "full")) {
        $filename = "S1A3_pose_S1_{0}_{1}.json" -f $camera, $model
        sumedipose --root "$root\raw" --camera $camera --subject S1 --repeat D2 --pose-baseline --model-task "$root\models\pose_landmarker_$model.task" --max-frames 24 --benchmark-json "$root\analysis\$filename"
    }
}
python -m sumedipose.calibration_strata --analysis-dir "$root\analysis" --output "$root\analysis\S1A3_calibration_stratified_pose_overview.json"
~~~

**Interpretation and safeguards:** calculate equal-*session*-weight descriptive Spearman correlations, never pool individual frames as independent observations. C1 is a view control alongside C6; the script also reports descriptive C6-minus-C1 MAE contrasts. Compare Lite vs Full **within the same 2D or 3D reference space**, not 2D projected-Vicon errors directly against 3D WCS errors. The high/low split at **5 px** of G65 epipolar error was chosen **post-hoc**, not as a clinical threshold. No inferential p-values, clinical accept/reject decision, exposure-time synchronization or training-set overlap audit are supplied.

The tool refuses to claim a complete five-session comparison when any required model/view report or valid knee-angle session measurement is absent. It rejects sample/model-SHA/24-image protocol inconsistencies. Reports contain aggregate scalar associations only, not RGB photos, per-frame poses or participant-level measurements. Output must remain outside this public Git checkout; existing output files are never overwritten.

## New-participant A3 squat repeat validation (same five calibration groups)

The original five-session calibration–pose exploratory result used **one anchor participant per calibration group**. We subsequently selected **two different participants within each of the same five groups**, and evaluated three A3 squat repeats (D1/D2/D3) per participant, both C1 frontal and C6 posterior views, using the **unchanged** MediaPipe Lite/Full model assets and 24 uniquely sampled RGB frames per recording. This makes 10 additional participants, 30 recordings, 120 aggregate model runs and 2,880 image inferences — but **zero additional independent calibration sessions**.

The analysis intentionally avoids pseudo-replication: weight the valid left/right knee-angle observations *within one recording*, then average across three repeats *within one participant*, then average across the two new participants sharing a calibration group. Associate only the resulting **five group means** with the five G65 (C6–C5) calibration Sampson residuals using *descriptive* Spearman rho. C1 serves as a view control; 2D projected-Vicon and 3D WCS knee-angle outcomes remain separate. The model predictions are **monocular**, whereas G65 is a **two-camera stereo** residual. A correlation is not a causal effect, independent 3D accuracy, or a medical quality threshold.

Only the local Windows analysis folder contains the preselected subject mapping, model outputs, and original ZIPs. Once those 120 reports exist and all SHA/sampling/identity checks pass, reproduce their aggregate analysis with:

~~~powershell
$folder = "D:\Research\PublicDatasets\SUMediPose\analysis"
python -m sumedipose.new_subject_validation --analysis-dir $folder --output "$folder\A3_grouped_repeat_validation_new_run.json"
~~~

Source code and synthetic tests may be published, but **never the per-subject report JSONs, original image archives, model weights, or detected coordinates**. The command fails closed on any missing/mismatched report or model SHA, and refuses to overwrite an existing report or write it under the public repository.

Exploratory results from the new participants (five original calibration groups, not ten independent sessions): C6 3D WCS knee-angle MAE vs G65 residual had **rho 0.60 (Lite)** and **0.50 (Full)**, compared with **0.90/0.70** in the anchor participants. This is positive but weaker evidence; C1 and 2D comparisons are not consistently specific to C6. The five-group sample, within-session participant sharing, model/Vicon landmark differences, lack of raw hardware exposure timestamps, and possible model-training data overlap prevent inference of clinical 3D accuracy. All three repeats are **squats**, not independent exercise types.

## A4 deadlift action-generalization pilot

Set --action A4 to run the same MediaPipe Lite/Full benchmark on a deadlift recording; the default remains --action A3. Only sample inspection and pose benchmarking support A4; timing/calibration audit commands remain A3-specific.

Example (use approved local storage paths):

    sumedipose --root DATASET_ROOT --action A4 --camera C1 --subject S1 --repeat D2 --pose-baseline --model-task LOCAL_MODEL_LITE_TASK --max-frames 24 --benchmark-json PRIVATE_A4_SUMMARY_JSON

Once all five matching calibration anchors (S1, S6, S11, S17 and S23) have A4 reports for C1/C6 and Lite/Full, generate the paired descriptive summary:

    python -m sumedipose.action_generalization --analysis-dir PRIVATE_ANALYSIS_DIR --output PRIVATE_SUMMARY_JSON

The A3 and A4 comparison is five paired calibration groups, not ten independent calibration trials. G65 C6-C5 stereo residuals are not direct evidence of monocular accuracy. Physical capture synchronization, Vicon-vs-model landmark equivalence and clinical knee-angle validity remain unverified. Original RGB ZIP, Vicon coordinates, local model assets and all participant-level reports are excluded from Git.

## Next validation: genuinely independent camera calibration sessions

The existing SUMediPose A3 and A4 results share five daily camera calibration sessions. Increasing participants, exercises, views or image-inference counts does not create new independent calibration evidence. For the prospective capture design, held-out C6-C5 stereo-quality measure, capture-synchronization gates, privacy and external dataset limitations, see [Independent calibration-session validation protocol v1](docs/independent-calibration-study-v1.md). No new calibration acquisition is claimed.

## A4/D2 3D Vicon angle-bias and record-index follow-up

The existing [A3/D2 frame-index and signed angle-bias pilot](docs/a3-d2-frame-joint-pilot.md) can also be run for deadlift via `python -m sumedipose.frame_joint_pilot --action A4`. See the [A4/D2 3D Vicon comparison protocol](docs/a4-d2-vicon-3d-bias-pilot.md) for the source-only coordinate definitions, frozen 24-frame comparison, private-output boundary and critical limitation: **a record-index shift is not a verified exposure-time alignment, and the model landmarks are not independent Vicon joint centres**.

## Experimental root-relative 3D position geometry (not MPJPE)

A strict `dup_num == factor` RGB-to-WCS mapping audit identified eight dropped-frame placeholder reference pairs in the historical five-anchor A3/A4 sampling. The [held-out A3→A4 root-relative six-joint diagnostic](docs/a3-a4-rootrelative-3d-holdout.md) fixes source frame selection, fits one similarity transform on A3 training frames, and checks frozen 3D **shape** discrepancy on held-out A3/A4. It does **not** validate absolute XYZ positions, real-world MPJPE, anatomical joint-centre equivalence, or physical Vicon-video timing; filled reports stay on the private research drive.

## A3 26-person / A4 anchor strict source-reference sensitivity

A frozen-image [Vicon source-frame correspondence recheck](docs/a3-26-a4-5-strict-vicon-reference-recheck.md) pairs each identical MediaPipe pose prediction against the original midpoint coordinate index and the `dup_num == factor` source-valid coordinate index. It recomputes descriptive G65-versus-angle-MAE correlations at the **same five historical calibration groups** and preserves raw/participant outputs on private local storage. This checks reference-index sensitivity, not independently synchronized capture timestamps, causal calibration effects, absolute XYZ MPJPE, or medical validity.
