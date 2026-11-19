# Test-time Vicon-free camera-relative 3D localization (experimental PnP)

**Status:** source-private experiment. Implementation and validation of **model-only test-time camera location estimates**; not a clinically validated full-world MPJPE(mm), nor a marker-free universal calibration system.

## Why this experiment differs from previous A3/A4 studies

Earlier [cross-person LOCO 3D shape evaluation](cross-person-a3-a4-3d-alignment-transfer.md) used Vicon GT hip centres on **each test frame** to root-centre the reference and AI-predicted skeletons. It measured geometry *after supervised reference centering*; it did **not** estimate the location of the person inside the actual camera's 3D space.

This experiment attempts to infer an entire six-joint camera-frame skeleton **without passing any test Vicon XYZ, hip-centre or bone length into the location predictor**. Ground truth from the original dataset is used only **after** the prediction to compute diagnostic discrepancies.

## Predeclared input and model

- 26 available A3 D2 participants and five A4 D2 anchors in the **five original calibration groups** (not 26 independent camera setups), views C1 front/C6 rear, frozen MediaPipe Lite/Full task hashes; 24 original source-valid RGB images per recording and original \`dup_num==factor\` correspondence.
- Source camera **intrinsics (K) and distortion** from official \`external_calibration.csv\` referenced matrices. C1 uses the **main C1 intrinsic of G12**; C6 uses the **main C6 intrinsic of G65**. CSV fixes which file to use; do not choose matrices or pose hypotheses by looking at held-out Vicon errors. The held-out camera's **pre-existing calibration K is provided**: this experiment does **not** generalize to an unknown uncalibrated camera.
- For each C1/C6 and Lite/Full, and for each of five leave-one-calibration-group-out folds, the **only training Vicon information** is the **hip-to-knee 3D segment length** in derived camera XYZ in the A3 **first 12** source-correct frames of one anchor person from each of the **four other** calibration groups. Fit a single numerical scale \`camera XYZ source units per MediaPipe metre\` as the **median of four equal-group medians**. The entire held-out group, all held-out A3 last-12 frames, and all A4 frames are excluded from this scale.
- On each held-out test frame, obtain MediaPipe's **image 2D landmarks (pixels)** and **pose_world_landmarks 3D** from the same RGB image. Pass these, the fixed **training scale**, and this camera's frozen **K/distortion**, to \`cv2.solvePnP\` EPNP. Solve per-frame image-only rotation + translation; explicitly require six finite landmarks, positive predicted joint depth and no numerical errors. **No test-time Vicon GT is passed into \`predict_camera_points\`.**
- Only *after* successful PnP, compare predicted camera-frame hip-midpoint/root and six-joint 3D positions to SUMediPose's **Vicon-derived camera XYZ**. Express Euclidean discrepancies as **% of test Vicon femur segment length**, not an independently verified distance in millimetres. The reference segment length is used only to **normalize the metric** after prediction. Count all attempted frames, valid six-joint frames, positive-depth PnP results and rejected PnP cases; do not hide bad outcomes by filtering on reference error or model self-reprojection.
- Separately audit source Vicon-derived camera XYZ projected with the selected K/distortion against the dataset's **Vicon-derived 2D projections**. This is an **internal calibration evidence check only**, not an independent RGB/Vicon registration test. Flag per-group p90 residuals exceeding a **predeclared 20 px** evidence-warning threshold without manipulating or excluding the test scores.
- Report group-first equal-weight **five calibration-group medians**, separately for A3's held-out last-12 frames in 26 subjects and A4's 24 frames in five anchors, for both model variants and both views. Each original calibration group is held out once per view/model; **new independent calibration groups = 0**.

Expected model-image inferences: **2,496 A3 + 480 A4 = 2,976**. New code does **not** use test Vicon hip centres or test GT bone lengths for location inference; having the evaluation reference available in the process solely permits ground-truth scoring.

## Why this is not validated MPJPE(mm)

1. **Metric camera units** are defined by source camera calibration, including the dataset's **nominal board-square scale**, not an independently measured metrological standard. Even if the source board square is nominally 60 mm, numerical source-unit differences should not silently be marketed as physical millimetre errors.
2. The **target group's own pre-existing camera intrinsic calibration** is used. This is not a full no-calibration domain transfer; earlier cross-group training determines only the PnP input scale.
3. RGB–Vicon **original exposure timestamps and clock synchronisation are unverified**. Prior source frame index correction does not establish physical exposure simultaneity.
4. **MediaPipe hips/knees/ankles** are different landmarks from Vicon's anatomically modelled **joint centres**, so the six-joint metric conflates point-definition and camera-space reconstruction errors.
5. The projected 2D image points and 3D world landmarks are from the **same pose network**. PnP self-reprojection can be small while real joint positions differ from Vicon, or vice versa. The source-projection check uses two representations derived from Vicon, **not independent hardware proof**.
6. This is a feasibility study of camera-root recovery **within one public dataset**; the model's potential training-data overlap and external generalisation are unknown. No clinical thresholds, therapeutic decisions, causal effects, external-study results or deployment authorisation follow from this experiment.


### Source calibration metadata inconsistency (pre-inference discovery)

A read-only audit of the released `external_calibration.csv` found a **conflicting source filename for S17/C6/G65**: one row associated with S17 camera 6 points at a **S23** calibration matrix filename, while a second S17/G65 row (camera 5, with main camera 6) points to an **S17** matrix filename. This is a **source-provenance ambiguity**, not permission to use test Vicon to choose the lower-error matrix. The frozen algorithm requires the candidate's source-calibration filename to identify **the same original calibration group**, with the expected main camera and camera-pair identifier. It refuses multiple group-consistent candidates; it does **not** select intrinsics by held-out 3D ground-truth fit. Results for C6/S17 remain conditional on this source-metadata choice until the source authors resolve the discrepancy; the finding must be reported even if the numeric PnP test completes.

## Windows local-only execution

Do not modify original RGB/WCS/CCS ZIPs, task files or a potentially dirty user worktree. Use a **dedicated clean code checkout**. All per-group/subject summaries remain on the approved research PC, outside GitHub, Second Brain, Drive and Notion. Only reviewed nonidentifying aggregate interpretations may leave.

\`\`\`powershell
$repo = (Get-Location).Path
$base = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path $repo "src"
python -m sumedipose.camera_pnp_no_test_vicon \`
  --root "$base\raw" --models-dir "$base\models" \`
  --analysis-dir "$base\analysis" \`
  --private-output "$base\analysis\camera_PnP_LOCO_no_test_Vicon_private_20261119.json" \`
  --aggregate-output "$base\analysis\camera_PnP_LOCO_no_test_Vicon_aggregate_20261119.json"
\`\`\`

The report uses **strict JSON**, refuses to overwrite existing files, and will fail closed if the frozen model hashes, source image count, calibration membership, selected source K or minimum viable six-joint data cannot be verified. The code includes no files exposing participant RGB, Vicon XYZ trajectories or model keypoints.

## Acceptance and interpretation

- [ ] Private experimental report and aggregate pass strict JSON parsing.
- [ ] Five LOCO held-out groups × C1/C6 × Lite/Full = **20** folds; 26 A3 people and five A4 anchors; model hashes and original sample selection unchanged.
- [ ] All four training calibration anchor groups supply at least six valid A3 first-12 frames; held-out group contributes **zero** reference XYZ to the scale learner.
- [ ] All test-time locations come from **MediaPipe RGB 2D+3D, training-group scale and target camera K/distortion only**; reference GT consumed only for scoring **after** inference.
- [ ] Independently recompute group-level median root and six-joint errors, source projection K diagnostics, PnP availability, fail rates and per-cell frame counts from private fold summaries; do not release per-participant results.
- [ ] Clearly label any C6 source-projection warnings and unknown physical timestamp/anatomical equivalence as unresolved measurement validity, not a valid clinical error.
- [ ] No nominal board-scale distances presented as independently verified **MPJPE(mm)** or as a medical-device performance claim.
