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
