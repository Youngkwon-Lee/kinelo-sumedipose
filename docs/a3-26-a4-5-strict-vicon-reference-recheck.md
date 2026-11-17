# Strict RGB-to-Vicon coordinate reference recheck: A3 26-person D2 / A4 5-anchor D2

**Status:** frozen local re-evaluation protocol; code does not itself certify completed inference, calibrated camera–Vicon timing or clinical accuracy. Original source RGB archives, per-person reports and Vicon coordinates remain only on authorized local research storage.

## Why this analysis is necessary

The source camera metadata's `path` can reference a JPEG also used by nearby Vicon coordinate records after a dropped camera frame. For an actual JPEG, the source-valid coordinate record is the **unique** record where `dup_num == factor`; the earlier exploratory baseline selected the middle record of each JPEG-path group. Incorrect source associations change ground-truth knee angles **without any change in model output**.

A bounded read-only preflight of all **52 A3 D2 recordings (26 participants × C1/C6)** and **10 A4 D2 recordings (five original calibration anchors × C1/C6)** found **16/1,248 A3** and **5/240 A4** old image–Vicon coordinate pairs whose selected index was not the source-valid one: **21/1,488 unique JPEG sample instances**. Every recording still has 24 uniquely matched images and the strict sample selects the **same JPEG filenames, in the same order**. The corrections can therefore be evaluated as **paired differences in ground-truth coordinates only**. Since two MediaPipe models are independently run on each image, exactly 42 model-image inference pairs can have changed reference labels, out of 2,976 expected model-image inference pairs. This count does not itself establish the numerical impact on old MAE/correlations.

## Fixed method, unchanged source and model

1. Reuse the original **A3 D2 26-person cohort** (original file availability excludes two people), its three disjoint compositions (**anchor5**, **other10**, **remaining11**) and original private subject-to-five-calibration-group mapping. Do not substitute the D1/D2/D3 repeat averages for D2. Keep the existing **A4 D2 five anchor** group as a separate action comparison.
2. Keep C1 front and C6 rear views, frozen **24 JPEG filenames per recording**, fixed single-image MediaPipe Pose Landmarker Lite/Full model SHA256 and minimum per-landmark visibility >=0.5. Both 2D image and 3D world landmarks come from each single model, not independent 2D vs 3D networks. All 124 model runs are performed anew for exact paired reference evaluation, not to retrain.
3. For each JPEG, use a **single inference output** to compute both original midpoint-index error and strict `dup_num==factor` index error against the appropriate Vicon reference: projected `xy` for 2D included hip–knee–ankle angle; WCS `xyz` for 3D included angle. Preserve missing detections and visibility filtering, and require a source-valid unique coordinate record; **fail closed** if a recording's fixed JPEG set changes.
4. Independently verify that rerun **original-reference** left/right visibility counts, pose-detection counts and side-specific MAE match the saved original report to the frozen tolerance. If not, **do not** infer a GT-index-only causal contrast. Original per-person outputs are never overwritten; new, checkpointed reports are held in a separate private analysis subdirectory.
5. Combine **left/right valid-angle-count-weighted MAE** per person and view/model/reference space, then average people equally *inside each of the same five calibration groups*. Report descriptive **Spearman rho of G65 stereo held-out calibration residual vs group-mean MAE**, original vs strict, for four A3 D2 compositions × C1/C6 × Lite/Full × 2D/3D = **32 rows**, plus the A4 five-anchor cohort's **8 rows** (40 rows total). Report observed changes in participant-median MAE, five-group leave-one-out sensitivity and detection counts as descriptive outcomes. Reproduce previous A3 cohort rho values from rerun's original reference before releasing new aggregates.
6. Exclude all per-person and per-frame inputs from public Git, Drive, Second Brain and Notion. Only publication-appropriate **nonidentifying** aggregate changes and explicit limitations may leave private Windows storage after review.

## Authorized local execution

Run on a new, clean code-only checkout of `kinelo-sumedipose`, **outside** the existing repository's user worktree, with all model archives and source data retained separately. The script checkpoints only aggregate *per-person report JSONs* in a dedicated private results folder. It refuses to overwrite them on retries and fails on conflicting identities; it writes the final cross-cohort aggregate only after all 124 reports and original replay checks pass.

```powershell
$repo = (Get-Location).Path
$base = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path $repo "src"
python -m sumedipose.strict_pairing_recheck `
  --root "$base\raw" --model-dir "$base\models" `
  --analysis-dir "$base\analysis" `
  --output-dir "$base\analysis\strict_source_pairing_20261117" `
  --max-new-reports 124
```

A bounded **one-report preflight** uses the same command with `--max-new-reports 1`; after its recorded replay passes, rerun to process the remaining 123. The first pass produces `incomplete_source_valid_recheck` and must **not** be presented as a completed 26-person study.

## Research boundary and decision gates

- [ ] 124/124 model report identities, model hashes, actual detection counts and original-reference metrics reproduce the previous frozen results.
- [ ] 2,976 model-image predictions attempted, correct source-valid labels applied, same fixed image names; 21 affected unique JPEG↔reference assignments (verify, do not assume).
- [ ] A3 4-cohort × 8 outcome and A4 anchor × 8 outcome **40** Spearman rho values / MAE contrasts recomputed; old A3 32 rho values matched to historical saved descriptive outcomes before interpreting strict changes.
- [ ] Changes in C6 3D WCS Lite/Full group correlations and Full remaining11 sign reversal assessed separately; never imply that changing ~1% of frame references guarantees sign stability.
- [ ] Independent calibration acquisition count is **still five historical groups, zero new**. The G65 stereo residual is not a proven cause of single-camera AI error.
- [ ] Physical capture timestamps, Vicon/model anatomical landmark equivalence, absolute world-camera coordinate registration and real-world MPJPE(mm) all remain **unverified**, irrespective of statistical changes.

This protocol uses an existing published dataset, no new participant filming or clinical research approval; future **new** filming and/or commercial reuse require separate ethics/license review. See [DATA_POLICY.md](../DATA_POLICY.md) and the [root-relative 3D shape pilot](a3-a4-rootrelative-3d-holdout.md).

## 2026-11-17 — completed Windows-local source-valid re-evaluation

**Execution:** On the authorized research PC, all **124 / 124 model reports** (A3 26 subjects × C1/C6 × Lite/Full = 104; A4 five anchors × C1/C6 × Lite/Full = 20) were successfully re-evaluated on the **same fixed original 1,488 unique RGB JPEG sample instances**, making **2,976 model-image inferences** and **2,949 pose detections**. Each saved original-reference side MAE, valid-side count and pose-detection count was reproduced before the strict report was accepted. Frozen Lite and Full model SHA256 values matched the original A3 26-person protocol; original reports and RGB/Vicon ZIPs were not overwritten.

**Source association audit:** **16/1,248 A3 + 5/240 A4 = 21/1,488** unique JPEG↔Vicon-record pairings had originally selected a dropped-frame placeholder coordinate index. Two model variants were run on the same images, so this is **42/2,976 model–reference pairings** at potentially changed labels. All **1,488 JPEG filenames and their order remain unchanged**. The strict source-valid correspondence requires a unique `dup_num == factor` record; no image or participant substitution occurred.

**Sensitivity results:** The study recomputed 4 disjoint/overlapping A3 D2 cohort definitions (historical 5 anchors, disjoint 10, remaining 11, all 26) and the 5-anchor A4 D2 results across C1/C6, Lite/Full and separate 2D projection/3D WCS spaces: **40 five-calibration-group Spearman rho comparisons**. The original four A3 cohort correlation sets (**32** rhos) reproduced the historical source analysis exactly. A separate, Windows-local independent verification using SciPy's `spearmanr` recalculated **all 80 original and source-valid rhos (80/80 match)** and the group-weighted participant MAE summaries.

**39 of 40 rho values were unchanged.** The sole change was **A3 anchor-five, C6 rear, MediaPipe Full, 2D projected Vicon reference: rho −0.30 → −0.40**. The maximum absolute movement of the *participant-MAE median* across the 40 summary rows was **0.2215°** (not a maximum individual-frame error).

| C6 rear, 3D WCS knee included-angle MAE vs G65 calibration residual | Original rho | Source-valid rho |
|---|---:|---:|
| A3 anchor 5 Lite / Full | +0.90 / +0.70 | +0.90 / +0.70 |
| A3 disjoint 10 Lite / Full | +0.60 / +0.50 | +0.60 / +0.50 |
| A3 remaining 11 Lite / Full | +0.50 / −0.60 | +0.50 / −0.60 |
| A3 all 26 Lite / Full | +0.90 / +0.50 | +0.90 / +0.50 |
| A4 anchor 5 Lite / Full | +0.70 / +0.90 | +0.70 / +0.90 |

The **Full-model rho sign inversion in the remaining 11-person A3 cohort persists** and is **not explained by the 21 corrected original coordinate indices**. The same five calibration sessions are still shared; this re-evaluation added **zero** new independent calibration groups. These findings support robustness **to this specific source-index repair only**, not a causal explanation, external reproducibility, clinical validity, 3D MPJPE(mm) or validated physical camera–Vicon exposure-time synchronization.

**Reproducible private outputs** (all remain in the existing Windows research analysis folder, no source/subject records in public Git): `strict_source_pairing_20261117/` contains 124 local per-subject/view/model aggregate report JSONs plus `strict_A3_26_A4_5_D2_aggregate_20261117.json` and `strict_A3_26_A4_5_D2_independent_scipy_readback_20261117.json`. Local synthetic suite was **85/85 passing** on code SHA `dcf7122da8dca0d71db23acc07548570da0aa293`, and the PR CI ran Python 3.10/3.12/3.13 successfully on that SHA; the final documentation-head CI and main-commit checks must be read separately.

**Interpretation:** The corrected index mapping removes a known labeling artefact while keeping pixel/model inputs fixed. It does **not** prove exact physical capture synchronization, or resolve differences between MediaPipe landmarks and Vicon joint centres. All numerical results remain exploratory, noncausal and unsuitable for clinical accept/reject thresholds.
