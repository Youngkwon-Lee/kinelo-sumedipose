# A4/D2 3D Vicon knee-angle comparison — signed bias and record-index sensitivity

Status: **A4 local exploratory pilot executed on 2026-11-17**; original prospective method and constraints are retained below. This document specifies the A4 follow-up to the already executed A3/D2 five-anchor [frame/joint pilot](a3-d2-frame-joint-pilot.md). It does **not** claim physical exposure-time synchronization or clinical accuracy.

## Scientific question

1. On the **same five source calibration groups** and using the same 24-image selection and MediaPipe model hashes, what is the signed model-versus-Vicon 3D knee *included-angle* bias during A4 deadlift?
2. How much does angular MAE change when the WCS coordinate **record index** is shifted by a predeclared `-12, -6, -3, 0, +3, +6, +12` records? Is zero shift best? In-sample shift selection is a **sensitivity test, not synchronization or a corrected timing estimate**.
3. Does the previously observed large C6 rear-view signed bias on A3 also appear for A4? Compare action-specific view/model cells as descriptive summaries only.

## Reference and model coordinate definitions

- Original WCS `xyz` contains Vicon-derived 3D hip, knee and ankle joint centres (`LHJC/LKJC/LAJC`, `RHJC/RKJC/RAJC`). The 3D Vicon reference is the included angle formed by each triplet.
- The camera `internal_data` `xyz` is also Vicon-derived, expressed in another coordinate system; close WCS-vs-camera 3D angles test source consistency, **not** two independent 3D ground-truth systems.
- MediaPipe `pose_world_landmarks` (hip 23/24, knee 25/26, ankle 27/28) are model-defined hip-centred 3D landmarks. Their included angle may be compared to the Vicon 3D triplet **as a geometric feasibility metric** after checking visibility, but the landmarks do not share identical anatomical joint-centre definitions.
- **Do not** directly subtract WCS XYZ positions and MediaPipe world XYZ positions or label the result millimetre 3D position error: origins, units, axes, and anatomical landmarks are not registered, and independently measured capture timestamps remain unverified. A valid positional study needs explicit camera/world-to-Vicon registration and landmark mapping, and separately validated synchronization. Two-dimensional projected angles must remain in a **separate** reference space from WCS 3D.

## Frozen local run

The original A3 reports already contain signed bias and record-index sensitivity and should be **reused**, not re-inferred or overwritten.

Run the A4 follow-up on the authorized Windows research PC only, with source ZIPs, Vicon coordinates, and model weights kept in private storage:

```powershell
$repo = (Get-Location).Path  # approved, clean kinelo-sumedipose checkout
$base = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path $repo "src"
python -m sumedipose.frame_joint_pilot --action A4 `
  --root "$base\raw" --model-dir "$base\models" --analysis-dir "$base\analysis" `
  --output "$base\analysis\A4_D2_frame_joint_pilot_5anchors_local_20261117.json" `
  --aggregate-output "$base\analysis\A4_D2_frame_joint_pilot_5anchors_aggregate_20261117.json"
```

Expected attempted images: **5 existing anchor subjects × 2 views (C1/C6) × 2 model variants × 24 unique images = 480 model-image inferences**, comprising 240 unique images evaluated by each model. These are **not 480 independent participants, frame pairs or calibration sessions**.

The `--action` default remains `A3` for backward compatibility; A4 is opt-in. WCS/camera record-index alignment is checked but physical timestamp synchronization is not established. A missing record or model hash mismatch must fail closed. The `--output` files contain per-anchor derived results and must stay outside public Git, Second Brain, Notion and Drive. The source archives must never be extracted, rewritten or published.

## Completion and interpretation gate

- [ ] Original A4 image and Vicon coordinate sources verified; model task SHA equals the frozen prior protocol.
- [ ] All 20 A4 view/anchor/model runs completed, matching 24 unique images and recording detected pose counts and missing valid angles.
- [ ] All 8 action/view/model/modality aggregate cells contain **five** source calibration groups; compare A4-vs-A3 **within** modality, not 2D-vs-3D MAE.
- [ ] Review signed bias, zero-versus-in-sample-best shift and view-specific patterns; report **direction and uncertainty** without claiming hardware synchronization, Vicon registration, causal camera effects or clinical ROM accuracy.
- [ ] Keep **historical independent calibrations = 5** and **new independent calibrations = 0** regardless of model/frame count.
- [ ] Publish only non-identifying, approved aggregate evidence plus private local source pointers through established candidate-review pathways.

The A3 historical result for the five anchor groups had a C6 WCS-3D model-vs-Vicon knee MAE median about **25.61° Lite / 26.69° Full**, absolute signed bias median about **20.26° / 23.49°**, and shift zero best or tied in only **one of five** A3 anchor recordings for each C6 model. This is a warning to scrutinize model anatomical geometry and record-index correspondence; it does **not** estimate a physical video/Vicon delay.

## 2026-11-17 — local execution receipt (descriptive, five existing groups only)

On the authorized Windows research PC, A4/D2 completed **20/20** recording×view×model runs, **480/480** attempted model-image inferences on 240 distinct source RGB images, and **453/480** pose detections. Source model assets were checked against the frozen model SHA256; original ZIPs were read in place, with no overwrite. The aggregate output and private detailed records were written **outside public Git**. A separate readback independently reproduced the aggregate **8/8** camera/model/reference-space cells from local reports and confirmed independence remains **five historical calibration groups, zero new groups**.

Five-anchor **3D Vicon WCS included-knee-angle** per-recording group-median MAE and group-median signed bias (MediaPipe minus Vicon), in degrees:

| View | Model | A3 MAE | A4 MAE | A3 signed bias | A4 signed bias | A4 positive/negative-bias groups |
|---|---|---:|---:|---:|---:|---|
| C1 front | Lite | 13.86 | 13.84 | +3.95 | −9.36 | 0 / 5 |
| C1 front | Full | 13.87 | 14.80 | +3.02 | −12.52 | 1 / 4 |
| C6 rear | Lite | 25.61 | 15.41 | +20.26 | +8.86 | 5 / 0 |
| C6 rear | Full | 26.69 | 15.08 | +23.49 | +4.45 | 5 / 0 |

For A4, shift index **zero was best or tied** in only **1/5 C6 Lite**, **3/5 C6 Full**, **1/5 C1 Lite**, **2/5 C1 Full** sessions. These are same-input, **in-sample** shifted-coordinate diagnostics. Their preferred index shifts do not establish a physical delay and must not be used as a synchronized correction. The view- and action-dependent signed-angle biases are noteworthy **exploratory effects**, but anatomy, source interpolation, occlusion, motion-range distribution and uncertain time correspondence remain alternative explanations. In particular, smaller A4 MAE is **not** proof that deadlifts are easier to measure clinically.

Local-only aggregate output: `A4_D2_frame_joint_pilot_5anchors_aggregate_20261117.json`; private per-anchor report: `A4_D2_frame_joint_pilot_5anchors_local_20261117.json`; independent local validation: `A4_D2_frame_joint_pilot_independent_readback_20261117.json`, under the existing authorized Windows analysis directory. No raw image, model weight, person-level value or patient information was committed or uploaded.

### Research outcome boundaries

- The experiment compares **included knee angles only**, not XYZ positional accuracy or clinical flexion-extension ROM.
- The camera `xyz` ground-truth representation comes from the same Vicon source as WCS, rather than an independently measured 3D ground truth.
- Five same historical calibration groups were reused for A3 and A4; exploratory angle bias patterns should be tested on new, actually independent setups and with anatomical landmark harmonization and capture-time ground truth.
- No p-values, statistical generalization, causal explanation, clinical cutoff, independent hardware time sync, or software-as-medical-device claim.
