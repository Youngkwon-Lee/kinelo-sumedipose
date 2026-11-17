# SUMediPose WCS ↔ derived-camera 3D coordinate parity and time-evidence audit

**Scope:** source-data representation integrity, not a clinical or independently measured 3D accuracy result. The camera `internal_data/xyz` was **derived from the Vicon WCS `xyz`**, as described in the [SUMediPose data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088771/). No 3D position error between two independent measurement systems is established by comparing these two representations.

## Three prerequisites for a genuine MPJPE(mm) experiment

1. **Physical time correspondence:** Obtain raw camera **exposure times** and the Vicon source clock/trigger or a validated independent sync event, including original start/end bounds, drop-frame metadata, interpolation uncertainty and residual drift. The authors describe a manually reviewed **foot-stomp post-processing interpolation**, which is valid dataset construction but not proof of independent common-clock capture synchronization. `frame_num`, `frame_num_w_drop`, EXIF/ZIP modification times and repeated JPG filenames **are not exposure-time evidence**. In this audited local release, WCS and derived camera JSON expose frame indices but not exposure timestamps or a common physical clock. The capture system was described as recording timestamp text files; their original availability and exact relation to the released samples is **unverified**, not assumed nonexistent.
2. **Physical coordinate system and metric scale:** The source paper specifies a **nominal 60 mm calibration-board square** and Vicon-world-to-camera transformations using the 5-LED axis wand. The camera-pair stereo `R/T` alone **does not register an independent MediaPipe 3D reconstruction** to Vicon WCS. Verify origin, handedness, scale, externally measured dimensions, camera-to-Vicon transformation and residuals on **independent held-out calibration objects** before calling anything true metric accuracy.
3. **Comparable anatomical joint definitions:** Vicon `LHJC/LKJC/LAJC/RHJC/RKJC/RAJC` are subject-specific joint centres derived from a marker-based processing model, while MediaPipe indices 23/25/27 and 24/26/28 are model landmarks. Defining an angular included-knee metric or fitting a subject-specific similarity does not make the 3D joint positions anatomically equivalent. Define comparable joints, calibration reference and acceptance uncertainty **before** reporting registered MPJPE(mm).

## An executable, fail-closed representation audit

The `sumedipose.derived_ccs_consistency` module uses only original Vicon WCS and its **already-derived** camera XYZ records (not RGB inference). For the **26 available A3 D2 participants** and **five A4 D2 anchors**, inspect C1 front and C6 rear, preserving the predeclared **24 unique JPG filenames** and matching each to exactly one source-valid `dup_num == factor` metadata record.

- Select six Vicon joint-centre triples, compare their WCS versus camera-derived 3D **segment lengths** and inspect the relative numerical scale.
- For each participant and camera, fit a **single** positive-scale orthogonal transform plus translation on **only the first 12 A3 images**. Explicitly compare an orientation-preserving transform (matrix determinant +1) to an axis-handedness-reversing orthogonal transform (determinant −1).
- **Freeze** both fits on the remaining 12 A3 images and, for the five anchors, transfer the fits to all 24 A4 images. No test A3/A4 points are used to choose a new transform. The reflected transform is a **coordinate-convention diagnostic, NOT a physically valid rigid camera rotation**.
- Report cross-group summaries of numerical representation discrepancy expressed in **original WCS-native units** and relative to each frame's Vicon femur length. A small discrepancy in this source-derived comparison measures **internal projection consistency** only; it is not an observed MediaPipe joint-position error, a real-world pose MPJPE(mm), or independently validated camera accuracy.
- Aggregate first within each existing calibration session, then across **five independently acquired historical calibration days**, rather than inflating calibration sample size to 26 participants or 62 recording views. No new independent calibration session is collected.
- Inventory WCS/camera record fields for physical timestamp keys. Absence of these keys in this **local selected release** does not prove the acquisition hardware produced no timestamps.

Run in a clean research-code checkout, with the original source files on authorized private storage:

```powershell
$root = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path (Get-Location) "src"
python -m sumedipose.derived_ccs_consistency `
  --root "$root\raw" `
  --analysis-dir "$root\analysis" `
  --private-report "$root\analysis\WCS_CCS_D2_26plus5_derived_private_20261117.json" `
  --aggregate-report "$root\analysis\WCS_CCS_D2_26plus5_derived_aggregate_20261117.json"
```

The module fails closed on unavailable strict image pairs, WCS/camera index mismatch, invalid six-joint coordinates, unregistered observation shapes, insufficient training frames, or existing output files. The private report contains per-person **summary metrics** and the group labels, but no raw RGB, 3D keypoints or fitted matrices. All private reports and originals remain outside public Git. A subsequent code version may independently rebuild the aggregate from that private local report and save a **separate versioned readback**, without rerunning heavy source processing or overwriting the prior report.

## 2026-11-17 exploratory source-derived coordinate finding

A 5-anchor pilot found that a fixed orientation-preserving proper rotation on A3 followed by A4 resulted in large WCS-to-derived-camera XYZ discrepancies; allowing **axis parity reversal (determinant −1)** dramatically reduced them, with an estimated numerical conversion near **1/60**.

The full 26-person A3 plus five-anchor A4 readback reports:

| Evaluation | Proper rotation, group-median % of Vicon femur | Axis reversal allowed, same metric |
|---|---:|---:|
| C1 front — A3 last 12 | 39.50% | **0.269%** |
| C1 front — A4 all 24 (anchors only) | 48.68% | **0.319%** |
| C6 rear — A3 last 12 | 39.85% | **0.289%** |
| C6 rear — A4 all 24 (anchors only) | 50.68% | **0.177%** |

- **Reflection-allowed fits improved the source-derived coordinate mapping for every inspected participant-camera case**: 26/26 for C1 and C6 A3, 5/5 for C1 and C6 A4, all **using Vicon-derived ground truth at both ends**. This is evidence of different numeric axis **handedness conventions** or an equivalent mirrored representation, not a clinical diagnostic discovery.
- Median WCS-to-derived-CCS six-joint segment-length ratio was approximately **60.008**; median fitted reflected scale was approximately **0.0166657 CCS units/WCS unit**, consistent with the published nominal 60 mm board-square conversion. Neither statistic independently certifies an SI-unit calibration or absolute Vicon error.
- No raw exposure timestamps or shared-clock event fields were found in the audited A3 WCS/derived-camera record schemas; source paper describes post-hoc stomp alignment. New independent calibration days: **zero**.
- **No result above evaluates MediaPipe predicted XYZ or repairs the previous root-relative MediaPipe→Vicon shape diagnostic.** Model predictions may use a different axis convention. They require an **independently locked model-to-reference parity and landmark protocol** and a held-out sensitivity test before interpreting the previous 3D shape discrepancy.
- There is no claim of physical-camera/Vicon independent registration, actual exposure synchronization, MPJPE(mm), clinically equivalent joints, new participants, p-values, or product readiness.

The original data license and research governance remain those recorded in [DATA_POLICY.md](../DATA_POLICY.md). Publish the code, synthetic tests and approved **cross-group** findings only; keep all source files, participant summaries, exact WCS/CCS transform matrices, model weights and image/video artifacts on approved private research storage.
