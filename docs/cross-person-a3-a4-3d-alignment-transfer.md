# Leave-one-calibration-group-out (LOCO): MediaPipe 3D alignment transfer to unseen people

Status: prospective analytical design, implemented as source-private computation; **do not infer data have been collected or outcomes evaluated from this document alone**. Only local-run evidence may close the gates below.

## Research question

Does an A3 squat **model-world → Vicon root-relative** numerical similarity alignment fitted from **different people and different camera-calibration sessions** transfer to unseen people from the excluded session, and from A3 to A4 deadlift? Unlike the earlier [per-person parity analysis](mediapipe-3d-parity-A3-A4-heldout.md), **no original 3D Vicon labels from the held-out person's camera-calibration group are allowed when fitting the transform**.

A separate previous [knee-angle *bias* transfer pilot](a3-a4-knee-bias-transfer.md) already examined a cross-group constant *scalar joint-angle correction*. This study instead evaluates **six 3D joint positions in hip-root-relative space**, with properly constrained global scale and orientation. They answer different questions and should not be conflated.

## Source and partitions

- Dataset: original SUMediPose 26 A3 D2 participants (S5/S10 have unavailable RGB, exclude rather than fabricate), **five existing calibration groups** from `raw/calibration/external/subject_map.json` with [4,4,6,6,6] people. A4 D2 is available and evaluated only in five anchor people (one per calibration group). **Historical independent calibration conditions = 5; new = 0**.
- C1 front, C6 rear. Frozen Google MediaPipe Pose Landmarker Lite and Full weights verified by SHA256 against prior A3 26 protocol. Use the same **24 unique original JPEG paths** per action/view/person, with source-valid `dup_num==factor` WCS indices and minimum visibility 0.5 for bilateral hip/knee/ankle landmarks. No original pixels, raw Vicon trajectories, per-frame keypoints or person-level outputs may enter public Git.
- For each model and view, run **five predeclared LOCO folds**. In a fold, exclude one entire camera-calibration group (all its people and the one anchor) from the fit. Training input = **A3 first 12 unique JPEG slots** from one anchor in *each of the other four groups* (nominal 48 image slots total, each requiring at least six valid 6-joint frames). Train outputs are Vicon-ground-truth-supervised.
- Fit two *single, shared* positive-scale orthogonal transforms using only those 4 calibration groups: proper rotation `det +1`, and reflection-allowing transform `det -1`. Choose one parity based on **training-set discrepancy only**, tie-breaking in favour of proper. Retain full held-out results for **both** parities; never choose the reported winner after viewing test labels.
- In the completely withheld group, evaluate **all participants' last 12 A3 slots** (26 people across the five folds); additionally, evaluate all **24 A4 slots** for its one A4 anchor. No withheld subject/frame, including A4, contributes to rotation/scale selection. Each held-out person can appear in only one test fold per camera/model.
- For a fair **anchor-only supervised reference**, retain the old per-person, same-identity A3 first-12 fitted proper/reflected scores on A3 last-12 and A4 from the [prior A3/A4 parity experiment](mediapipe-3d-parity-A3-A4-heldout.md). Label these an *in-person supervised upper reference*, **not** a fair deployed camera-only baseline. Do not claim every fitted method is comparable on the 26 participants.
- The evaluation uses Vicon GT hips per test frame to centre **each** target 3D skeleton; model landmarks use their own hip midpoint. Thus it checks **scale-and-axis normalised root-relative 3D shape only** (six-joint average Euclidean distance / Vicon femur length × 100), **not global translation, physical camera registration, absolute MPJPE(mm), clinical 3D validity or model-training dataset independence**.

## Execution on authorized Windows research storage

Run only from a clean code checkout. All source files and private person-level summaries remain on the authorized Windows research drive. Predeclared output filenames are new and protected against overwrite.

```powershell
$repo = (Get-Location).Path
$root = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path $repo "src"

python -m sumedipose.cross_person_3d_transfer `
  --root "$root\raw" --model-dir "$root\models" `
  --analysis-dir "$root\analysis" `
  --private-output "$root\analysis\cross_person_LOCO_3d_private_20261118.json" `
  --aggregate-output "$root\analysis\cross_person_LOCO_3d_aggregate_20261118.json"
```

Input inference expectation: **26×2 cameras×2 models×24 A3 = 2,496** and **5×2×2×24 A4 = 480**, total **2,976** model-image inferences. An inference is not an independent subject or calibration session. The code verifies 26-person cohort membership and original model SHA256, refuses missing core frames, calibrations or output collisions, and never silently imputes missing ground truth. Output only **five-calibration-group weighted** medians across participants within group, with 8 camera/model/evaluation-condition cells and full test/parity comparison.

## Acceptance gates

- [ ] 4 other calibration groups' anchor A3 training labels are the only observations used for each fold's orientation/scale and train-only parity decision.
- [ ] All 5 holdout calibration groups evaluated, A3 26 people and A4 five anchors, both models/views, exact 24 source-valid unique JPEGs, sufficient visible six-joint A3 last-12/A4 frames.
- [ ] Existing A3/A4 anchor pose detection and source pairing counts replayed; same SHA-locked model assets; *no per-frame model positions saved to Git*.
- [ ] Independently recompute 8 equal-group-weighted aggregate cells and test-anchor per-person supervised reference comparisons from private per-fold summaries.
- [ ] Report whether transfer gap widens versus per-person A3-fitted baselines; do not treat lower/higher percent as clinical accuracy; note cohort, site, action and model limitations.
- [ ] Confirm historical independent calibrations 5 / newly acquired calibration sessions 0. Physical capture-time sync, MediaPipe↔Vicon landmark anatomy and metric location error **unverified**.

## Actual local experiment and independent group readback — 2026-11-18

**Run:** all five predeclared leave-one-calibration-group-out folds per camera/model completed on the approved Windows research PC: **20 held-out model×camera×calibration fold assessments**, A3 D2 **26 distinct available participants**, A4 D2 **five original anchors**, **124 held-out participant×camera×model×action evaluations**, **2,976 image-model inferences**, **2,949 detected poses**. Frozen original MediaPipe model SHA256 assets and **source-valid RGB↔Vicon indices** were used; the original data and prior reports were not rewritten. Within each fold, all held-out people's images/GT were excluded from fitting the A3 rotation/scale and selecting parity.

Five original calibration-group equal-weight median **relative six-joint discrepancy (% of Vicon femur length)**. The reported `training-selected` parity was decided from four other groups' A3 **training images only**, never from the test-set scores.

| Camera and MediaPipe model | A3 held-out 26 people, training-selected | A4 held-out five anchors, training-selected | Number of folds choosing reflection in training |
|---|---:|---:|---:|
| C1 front Lite | **21.36%** | **21.26%** | 5 / 5 |
| C1 front Full | **21.27%** | **21.79%** | 5 / 5 |
| C6 rear Lite | **30.23%** | **35.35%** | 4 / 5 |
| C6 rear Full | **25.30%** | **27.14%** | 5 / 5 |

Full proper/reflected medians (train-only parity selection may not match the better **held-out** parity):

| Camera/model | A3 proper / reflected | A4 proper / reflected |
|---|---|---|
| C1 Lite | 41.95% / **21.36%** | 36.20% / **21.26%** |
| C1 Full | 42.35% / **21.27%** | 37.33% / **21.79%** |
| C6 Lite | 31.88% / **29.97%** | **33.39%** / 34.15% |
| C6 Full | 40.12% / **25.30%** | 34.54% / **27.14%** |

**Paired matched-anchor transfer penalty, same held-out A3/A4 test frames and same parity as the LOCO training-selected parity:** median of five anchor-level LOCO score minus that anchor's own A3-first-12 supervised fit score, in **percentage points of femur-relative discrepancy** (positive = cross-person transfer has larger discrepancy).

| Camera/model | A3 anchor subset, five | A4 anchor subset, five |
|---|---:|---:|
| C1 Lite | +1.95 pp | +2.82 pp |
| C1 Full | +2.28 pp | +2.13 pp |
| C6 Lite | +1.50 pp | +0.44 pp |
| C6 Full | +4.02 pp | +1.44 pp |

The A3 26-person group medians above and the **personal A3-fitted five-anchor** values belong to **different subject compositions**; only the **paired five-anchor penalties** compare the *same people and test frames*. Individual anchors can sometimes outperform the personal supervised fit when trained from other groups; a positive group median does not mean everyone worsened.

**Independent readback:** the private reports were reloaded under **strict standard JSON validation**. In 8/8 camera/model/evaluation cells, a separate calculation from all 20 model/view/group fold records reproduced the group medians, participant counts, per-fold **training-only** parity choices, valid-frame counts, original model asset SHA256 and paired five-anchor transfer penalties. The first run had inadvertently appended literal escaped newline characters to two private JSON outputs; both **original byte streams were backed up** on the same private research drive, and the sole malformed suffix was removed without altering any reported numerical values. The new source code now writes **strict JSON with a real terminal newline**, fails for non-finite values, and refuses overwrite; regression tests cover both cases. **No raw RGB, Vicon XYZ coordinates, per-frame model landmarks or private fold results were sent to GitHub.**

Private outputs, on approved Windows research storage only:

- `cross_person_LOCO_3d_private_20261118.json` (individual experimental fold summaries, not publicly shareable)
- `cross_person_LOCO_3d_aggregate_20261118.json` (cross-group summaries)
- `cross_person_LOCO_3d_readback_20261118.json` (independent aggregate arithmetical check)

### Scientific interpretation

A group-trained 3D shape alignment is **numerically applicable** to a held-out calibration group and held-out people, but its loss of accuracy versus a per-person supervised fit and the C6 Lite parity dependence make a **universal plug-and-play physical 3D calibration unjustified**. Model training overlap with the public dataset is unknown. All evaluation frames still use **Vicon GT hip-centre rooting**, five existing acquisition/calibration groups, and **unsynchronized original hardware clocks**; no real-world MPJPE(mm), clinical kinematics, independent dataset deployment, causal conclusion or new calibration sessions are established. The **next distinct question** is whether the model can produce comparable anatomical joint positions without test-frame Vicon GT and with independently verified camera/Vicon metric registration.
