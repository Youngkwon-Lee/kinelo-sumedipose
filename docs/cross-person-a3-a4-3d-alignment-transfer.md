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
