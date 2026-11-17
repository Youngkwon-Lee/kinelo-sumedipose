# MediaPipe world landmark parity — frozen A3 train → held-out A3/A4

Status: **exploratory five-calibration-group local experiment, executed 2026-11-17**. This is a **root-relative six-joint *shape discrepancy* diagnostic**, not physically registered MPJPE (mm), anatomical joint-centre equivalence, physical exposure-time sync or clinical validation.

## Question

When **MediaPipe 3D pose_world_landmarks** (hip-centred model coordinates in metres) are compared against **Vicon WCS** hip/knee/ankle 3D joint centres after hip-root centering, does a **fixed coordinate handedness reflection (det = -1)** yield a smaller held-out shape discrepancy than a **proper rotation (det = +1)**? Is any advantage stable from squat (A3) to deadlift (A4), across front C1/rear C6 and MediaPipe Lite/Full?

The prior [WCS↔derived camera XYZ audit](wcs-ccs-coordinate-parity-and-mpjpe-gates.md) found a reflected-axis convention between **two representations of the same Vicon data**. That finding cannot be applied automatically to MediaPipe, because the model outputs and anatomical landmark conventions differ. This experiment checks both parities **directly against the model predictions**.

## Frozen execution design

- Existing original **five calibration anchors** (no new calibrations), A3 D2 and A4 D2, cameras C1 front/C6 rear, Lite/Full models with frozen SHA256 from the earlier 26-person A3 protocol. Original 24 unique RGB images per camera/action/group and `dup_num==factor` source-corrected WCS references; no re-training.
- For each **participant × camera × model** combination, generate MediaPipe landmarks independently for A3 24 and A4 24 images. Require six visible bilateral hip/knee/ankle landmarks (visibility ≥0.5). No per-frame predictions or raw Vicon keypoints are saved.
- Compute **Vicon-GT hip centre** and **model hip-landmark midpoint** separately and subtract them on *every* evaluation frame. This GT-dependent operation means the evaluation does not measure camera-relative absolute translation or any independently registered world position.
- Fit a **single fixed positive scalar and orthogonal matrix** on the first **12 A3 images only**, under either determinant **+1** (proper rotation) or determinant **-1** (numeric handedness-reversing reflection). The same model training images are used for both candidates; **never** fit or select a new transform using withheld test images.
- Freeze the two alignments, apply to the **last 12 A3 images** and all **24 A4 images** for that same participant/view/model. Compute the mean six-joint Euclidean shape discrepancy divided by mean Vicon GT hip-to-knee segment length, expressed **as a percentage**. Each source camera/view/model/participant gets its own A3-fitted transform, so the experiment **does not** demonstrate cross-person or cross-camera calibration transfer.
- Require the **det=+1 output to reproduce** the previous local [A3/A4 holdout](a3-a4-rootrelative-3d-holdout.md) result and detection counts before accepting a new comparison. Do not change the original source data, frame selection or legacy reports.
- Aggregate across **five existing calibration groups, with equal weighting**. Report both methods regardless of which performs better; parity is a tested coordinate convention, **not** a single universal axis correction to deploy.

## Actual local results — 2026-11-17

**20 anchor/camera/model combinations; 960 attempted model-image inferences, 933 detected poses.** All 20 prior proper-rotation results independently reproduced; the aggregate was independently recalculated for **8/8 cells**. In all rows below, values are **median five-group root-relative six-joint shape discrepancy as percent of Vicon femur length**, not millimetres; lower is smaller.

| Camera | Model | Evaluation set | Proper rotation (+1) | Reflection allowed (-1) | Reflection better, groups |
|---|---|---|---:|---:|---:|
| C1 front | Lite | A3 withheld 12 | 37.28% | **21.38%** | 5/5 |
| C1 front | Lite | A4 all 24 | 43.29% | **19.74%** | 5/5 |
| C1 front | Full | A3 withheld 12 | 38.82% | **21.21%** | 5/5 |
| C1 front | Full | A4 all 24 | 44.74% | **20.58%** | 5/5 |
| C6 rear | Lite | A3 withheld 12 | **26.71%** | 27.83% | 3/5 |
| C6 rear | Lite | A4 all 24 | **29.95%** | 33.71% | 2/5 |
| C6 rear | Full | A3 withheld 12 | 35.52% | **25.25%** | 5/5 |
| C6 rear | Full | A4 all 24 | 35.09% | **27.91%** | 4/5 |

For C6 Lite A3, 3/5 groups individually favoured reflection, but the **median** proper result was smaller; these are distinct statistics and need not have the same direction. Across all 20 participant×view×model combinations, the *preference for reflection versus proper* was unchanged between A3 and A4 in **18/20** combinations. In the A3 held-out sets reflection was preferred in **18/20**, versus **16/20** A4 sets.

**Finding:** Front view C1 consistently benefits from reflection on this supervised, root-centred shape metric. Rear C6 is **not universally mirrored**: both the model variant and movement action matter. This rejects a simplistic “globally flip MediaPipe 3D XYZ to fix accuracy” explanation. These are **not** evidence that the model's world landmarks are intrinsically mirrored; the optimal transform can reflect camera viewpoint, training subset, different reference joint definitions and other modelling confounders.

## Original data, verification and clinical boundaries

- All original RGB/JPG, Vicon WCS records, model weights, fitted transforms, participant-level aggregates and any per-frame keypoints remain **on the authorized Windows research drive only**. Code, synthetic tests and cross-group aggregate findings may be shared under the [data policy](../DATA_POLICY.md). No actual participant image or raw trajectory is part of the publication.
- Private local analysis directory: `D:\Research\PublicDatasets\SUMediPose\analysis\mediapipe_parity_A3_A4_20261117`; generated 20 `parity_*.json` per-session summaries, `mediapipe_A3A4_parity_aggregate_20261117.json`, and the independently recalculated `mediapipe_A3A4_parity_readback_20261117.json` receipt.
- Validation: **95/95** Windows Python synthetic tests on initial implementation, model hashes matched the frozen original A3 dataset protocol; original proper-rotation numerical results **20/20 exactly reproduced**, aggregate arithmetic **8/8** independently checked, inference and detection counts equal the existing same-frame benchmark.
- **Historical calibrations = 5; new independent calibrations = 0**. No participant or frame is counted as an independent camera calibration. Without direct physical exposure timestamps, camera/Vicon world registration, anatomical joint-centre harmonization and externally measured metric units, the output is **not physical MPJPE(mm)**. This experiment does not validate clinical 3D accuracy, causal mechanisms, universal parity, commercial suitability or an autonomous treatment decision.
- The private Windows run used implementation head `5e07425b321e2f4764302057c7929a5dc2e8c043`. The later change to the audit field's ASCII name does not alter calculations; final code-head Python CI and merge checks must be separately confirmed.
