# SUMediPose S17/C6 stereo calibration CSV provenance — 2026-11-20

Status: **read-only source audit and local predeclared sensitivity experiment complete**. These are *internal Vicon-derived representation consistency* and exploratory MediaPipe-to-camera-frame PnP sensitivity tests, not an independent optical-calibration certificate, verified MPJPE(mm), hardware synchronization, or clinical validation.

## Correction to prior source-file selection

The earlier [test-time Vicon-free PnP study](camera-pnp-without-test-vicon.md) required the file name of every calibration matrix to match the calibration group's subject identifier, effectively selecting the **S17-named G65 file** for **S17/C6**, even though the dataset's own \`external_calibration.csv\` reports:

- **S17, source \`cam=6\`, stereo \`main=6\`, pair G65:** explicitly references a **S23-named** calibration matrix.
- **S17, source \`cam=5\`, stereo \`main=6\`, pair G65:** explicitly references an **S17-named** calibration matrix.

The source authors' code stores stereo \`mtx_A\` and \`dist_A\` with respect to the **main camera** (see [published extrinsic calibration script](https://github.com/chris-schreuder/SUMediPose_pipeline/blob/main/code/calibration/extrinsic.py)). A CSV row for the actual requested camera must be preferred to a companion-camera row, rather than forcing the filename's original calibration-group identifier. The anomalous cross-group path may represent deliberate reuse, a post-processing substitution, or a metadata error; **which physical capture was the authoritative calibration still requires confirmation from the original source**.

### Non-independent provenance check

Use the *same source-valid* \`dup_num == factor\` original JPG selection and compare how well each original CSV row's \`mtx_A/dist_A\` reprojects the Vicon-derived camera \`xyz\` back into the **also Vicon-derived** projected \`xy\` data for the original group.

| S17/C6 original dataset | Explicit \`cam=6\` row's S23-named matrix | Companion \`cam=5\` row's S17-named matrix |
|---|---:|---:|
| A3 squat, six existing S17-group participants, 24 frames each, 6 joints | **0.607 px** pooled median | **9.410 px** pooled median |
| A4 deadlift, original S17 anchor, 24 frames, 6 joints | **0.525 px** pooled median | **9.476 px** pooled median |

The same source-internal comparison had A3 p90 ≈ 5.85 px vs 10.56 px and A4 p90 ≈ 4.42 px vs 10.52 px. This is strong provenance evidence for the CSV's explicit **camera 6 source row**, **not** independent evidence of true optical re-projection accuracy or physical 3D ground truth. **The PnP algorithm never selects a matrix based on these test-reference projection residuals.** All source files and matrices stay private.

## Code and prior experiment impact

The deterministic source selector now prefers **the CSV row where \`cam == main\`** and the correct stereo pair, with a companion-camera fallback only when no direct row exists. Multiple distinct eligible paths for that role **fail closed**. Cross-group filenames are **flagged in returned provenance metadata** rather than silently rejected. S17's existing camera 6 matrix mismatch is therefore preserved as a source-integrity warning, while the explicit S17/C6 row determines which source matrix the published dataset associates with that camera. *The held-out camera's original intrinsic K is still used; this is not a no-calibration deployment.*

The earlier 26 A3/5 A4 [PnP pilot](camera-pnp-without-test-vicon.md) remains preserved and identified as using the prior **filename-group rule**. A bounded **S17/C6-only** sensitivity run re-inferred **336 model-image pairs, 334 detected poses**, using the previous frozen model SHA256, source JPGs and the old **four-other-group A3 training scale**. It explicitly reproduced the previous S17 group PnP reports under the old \`cam=5\` row before comparing them to the \`cam=6\` source row. **No new model training or regression to held-out GT for calibration selection.**

| S17/C6 six-joint camera 3D discrepancy (% of Vicon femur) | Old filename choice | Source CSV \`cam=6\` choice | Difference (pp) |
|---|---:|---:|---:|
| A3 squat Lite | 65.0741% | 63.7167% | −1.3574 |
| A3 squat Full | 62.6951% | 60.9924% | −1.7027 |
| A4 deadlift Lite | 148.7936% | 145.2120% | −3.5816 |
| A4 deadlift Full | 97.7695% | 94.2977% | −3.4718 |

The **five historical calibration-group equal-weight median remains unchanged** for all four affected **C6** camera×model×action cells:

- A3 C6 Lite **82.7442%**, Full **70.8621%**
- A4 C6 Lite **108.4644%**, Full **82.0717%**

Other camera groups and C1 cells were **unchanged and not re-inferred**. The C6 source-intrinsics warning does **not disappear**: the previously reported maximum group projected-reference p90 (roughly 81–83 px) remains attributable to other groups. Therefore **repairing the S17 source-row selection does not resolve the general camera-frame PnP position error**.

The source CSV role alone does not determine whether the supplied K/distortion is physically *correct* for the optical recording; the source owners' original calibration provenance needs separate confirmation. No true physical MPJPE(mm), clinical accuracy, or new independently calibrated sessions have been established.

## Sync evidence availability

The [published synchronization code](https://github.com/chris-schreuder/SUMediPose_pipeline/blob/main/code/gen_kpt_data/correct_cam_sync.py) refers to original timestamp and foot-stomp source artifacts:

- \`data/times/all_times_w_drops.csv\`
- \`data/stomp_sync_corrected_manual.json\`
- \`data/stomp_sync_refine_manual.json\`
- \`data/stomp_3d_manual.json\`

None of these four expected **source-pipeline artifact paths** is present in the inspected local public dataset release. The resulting WCS and camera JSON provide frame indices, not the raw exposure-times-with-Vicon-clock evidence. This means **physical capture synchronization remains unverified in our local data**, not that the authors never recorded timestamps or that their published post-processed foot-stomp alignment was absent.

## Reproducibility, private outputs and acceptance

On the approved Windows research PC, a clean dedicated code checkout executed **148 synthetic-only tests successfully**, and the S17/C6 candidate comparison completed. An **independent readback reproduced all four camera/model/action S17 group values** and **eight old/source five-group median calculations** from the saved private reports; the two original model hashes and the previous S17 group results under the paired-camera candidate were checked. Original image ZIPs, model weights, source CSV/NPZ and previous PnP local results were **not modified or overwritten**.

All actual source data and private model/participant summaries remain under the authorized research PC analysis directory:

- \`S17_C6_source_calibration_role_PnP_private_20261120.json\` (source-comparison per-person summaries)
- \`S17_C6_source_calibration_role_PnP_aggregate_20261120.json\` (nonidentifying source-sensitive aggregates)
- \`S17_C6_source_calibration_role_PnP_readback_20261120.json\` (independent arithmetic and source-role readback)

**Decision gates remaining:** source-owner confirmation of the S17/C6 matrix lineage and actual camera 6 intrinsics, independent camera/Vicon exposure-time/coordinate calibration, held-out physical checkerboard/wand accuracy, and harmonized MediaPipe/Vicon anatomical landmark definitions before making any metric mm or clinical 3D pose accuracy claim. Old benchmark results should be cited with their exact code/matrix-selection protocol, not silently revised.
