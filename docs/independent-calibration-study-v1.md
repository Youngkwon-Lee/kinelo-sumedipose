# Independent calibration-session validation protocol v1

Status: **prospective design / not executed** (2026-11-16). This is a code-only/public planning document, not a preregistration with a registry, scientific approval, study result, or evidence that new independent sessions exist.

## Research question and current evidence

Does a *calibration-quality measure obtained from independent held-out calibration images* associate with monocular pose-estimator knee-angle error across **new, independently acquired camera calibration sessions**, and does the association persist across the A3 squat and A4 deadlift actions and C1 front/C6 rear views?

The [SUMediPose paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088771/) describes five days of daily camera calibration, identified by S1/S6/S11/S17/S23 session tags. The source Dataverse [doi:10.7910/DVN/GRRROM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GRRROM) exposes five corresponding external calibration frame ZIPs. A3 D2 26-participant analysis and A4 D2 five-anchor analysis reuse these **same five calibration groups**. The current independently calibrated-group count is **5**; new participants, actions, repeats, frames and inference runs do not increase it.

A3 26-person C6 3D WCS knee included-angle MAE versus G65 stereo Sampson residual had descriptive Spearman rho +0.90 (Lite), +0.50 (Full); the Full model's separate 11-person subset had rho -0.60. The **five paired anchor participants**, not the 26-person cohort, yielded A3 rho +0.90/+0.70 and A4 rho +0.70/+0.90 (Lite/Full). C1 A4 showed negative correlations. These are hypotheses to challenge, not verified clinical thresholds or causal effects. [Code-only A4 pilot](https://github.com/Youngkwon-Lee/kinelo-sumedipose/pull/12).

## Two distinct validation streams

### Stream 1 — external-dataset *domain transfer* (not automatically independent calibration)

- [Fit3D](https://fit3d.imar.ro/fit3d) offers 4 RGB views, Vicon-aligned 3D skeletons and diverse fitness actions. Check **approved non-commercial research-use license** ([terms](https://fit3d.imar.ro/legal)), current rights, data access, exercise availability, per-recording camera metadata, calibrated setup/session boundaries, frame alignment and anatomical joint mapping *before any download or analysis*. Do not use its data for a commercial product or redistribute it. **Unknown calibration-day identifiers must not be invented from subject or camera IDs.**
- [TotalCapture](https://cvssp.org/data/totalcapture/) offers multi-view synchronized RGB/Vicon/IMU and requires institutional research access and a non-commercial/no-redistribution agreement. Its activities and calibration geometry differ; do not assume it has matching A3/A4 or G65.
- External-dataset results are a **separate estimand** (cross-domain model generalization). Different camera geometry or unavailable independent checkerboard holdout means its metric cannot simply be pooled with SUMediPose G65 Sampson residuals. Verify compatibility explicitly.

### Stream 2 — new prospective independent calibration sessions (primary path)

The minimum comparable configuration includes **C1 (front), C6 (rear), C5 (rear oblique stereo partner) and a separately synchronized Vicon reference**, not just two RGB cameras: G65 specifically refers to the C6–C5 stereo pair. If capture uses a different geometry, assign a new quality-metric name and keep the analyses separate. Other cameras are optional; use fixed hardware/lens/resolution settings where possible.

For every *new calibration session*:

1. Record a de-identified session code, date, site/rig version, camera geometry, rig remount and setup-reset status. One independent unit means one genuinely new calibration acquisition event, not just a new frame, exercise, participant, file name, model run or software recomputation.
2. Acquire intrinsic calibration and stereo C6–C5 images, with a verified metric checkerboard scale (SUMediPose source uses 60 mm squares). **Freeze** calibration fit images/parameters before choosing separate held-out checkerboard images; do not tune on the holdout or select the best ROI or model after seeing outcomes.
3. Save intrinsics/distortion, C6–C5 extrinsics, frame-level holdout diagnostics, detection coverage, source checksums, software/code versions and a bounded calibration reprojection/Sampson-quality estimate. Distinguish stereo correspondence residual from absolute world/Vicon registration accuracy.
4. Acquire camera exposure timestamps and a shared-clock, trigger, or independently measurable synchronization event with Vicon. A foot-stomp or filename/frame-number correspondence alone is **not** proof of physical synchronization. Record uncertainty and failure explicitly.
5. With ethical/institutional permissions and research participant consent, collect A3 and A4 in that session using predeclared front and rear views and unchanged MediaPipe Lite/Full model assets. Sample 24 **unique** RGB frames per participant/action/camera/model, matched across model variants, and preserve visible/missing landmark counts; never manufacture missing frames.
6. Keep all originals, actual calibration frames, timestamps, source manifests, participant identifiers and per-person pose reports on approved **private local research storage**. Public Git contains **only synthetic templates, code and non-identifying, appropriately approved descriptive summaries**.

A useful **feasibility target (assumption, not a power calculation)** is at least **five newly acquired independent calibration sessions**, each containing A3 and A4 paired on the same acquisition day with at least one consenting participant. More independent sessions will be needed for dependable inference; determine a defensible target separately by feasibility and precision/power analysis. If multiple participants share a calibration, aggregate them **within session** rather than claiming multiple independent calibrations.

## Frozen analysis before prospective data collection

- **Primary outcome:** for C6 rear and a chosen predeclared model, participant-level average 3D *included knee angle* MAE versus a harmonized, independently held-out 3D Vicon joint-center reference; average left/right using valid-angle counts, then participants equally within each calibration session. This is **not** clinical knee flexion or absolute clinical ROM.
- **Primary explanatory variable:** session-level **held-out** C6–C5 G65 Sampson residual (pixels), computed from an independent board-image set using unchanged fit matrices; document sample count and detector failures. If camera geometry or quality definition differs, do not pool with historical G65.
- **Analysis unit:** independent calibration session, not participant, camera, 24-frame sample or image-model inference. Report number of independent new session calibrations, participants within sessions, missingness and provenance.
- **Primary descriptive association:** Spearman rho of held-out stereo-quality residual versus session-mean C6 3D MAE, with sign and uncertainty reporting appropriate to the *actual* number of independent sessions. No claim of significance for exploratory n=5. Retain session-level leave-one-out robustness and report individual group contributions privately.
- **Secondary, explicitly exploratory:** Full vs Lite; A3 vs A4 paired session difference; C1 view as a non-specificity control; 2D projected reference separately from WCS 3D; subgroup/subject-composition sensitivity. Do not select the best positive correlation post hoc.
- **Gates:** report an analysis as incomplete rather than successful if calibration session identity, held-out quality evidence, paired exposure synchronization, reference joint definition, frozen model hash or sufficient unique frames is unavailable. Descriptive exploratory results can be reported with limitations, but **clinical validity and synchronization remain unverified** until supported by direct evidence.

## Evidence and execution checklist

- [x] Audit existing SUMediPose published calibration-day count: **5**; no new independent session was identified in that source inventory.
- [x] Separate external-domain evaluation from independent calibration-session validation.
- [x] Define planned acquisition, reference, data governance and primary/secondary analysis; **prospective protocol only**.
- [ ] Verify Fit3D/other external datasets' permitted use and actual session-level calibration provenance, then decide access.
- [ ] Obtain required institution/ethics/consent approvals for *new* participant recording.
- [ ] Acquire and independently verify at least one **new** rig calibration and held-out session before counting n=6; expand across days/setups based on the approved design.
- [ ] Record shared-clock evidence or a predeclared independent sync test; freeze calibration and model hashes.
- [ ] Execute blinded/locked action-model analyses; report same-day paired comparisons and group-wise precision.
- [ ] Review/approve any claim of robustness or clinical relevance separately; this document alone does not authorize deployment.

For a new locally completed session, start from `docs/independent-calibration-session-template.json`, keep the **populated** copy private, and validate independently before updating group counts. **Never commit filled individual/session-level manifests to the public repository.**

Research governance: see [DATA_POLICY.md](../DATA_POLICY.md).

## Privacy-preserving preflight (synthetic and future private records)

Use the code-only readiness gate before any new calibration session can be considered for an **independent** calibration count. Its current blank example should **fail closed**:

```powershell
python -m sumedipose.session_preflight --session docs/independent-calibration-session-template.json
```

This command returns `status=blocked`, an ordered list of *gate names only*, and `new_independent_calibration_sessions_added=0`. It intentionally exits with status 2 for an incomplete manifest. It checks equipment availability confirmation, a new rig-calibration record, C1/C5/C6 geometry and G65 compatibility, 60-mm metric board, frozen fit/held-out board frames, timestamps and Vicon reference, paired A3/A4, model/24-image protocol, and required permission assertions. The **template defaults to all gates incomplete**, and its contents are synthetic only.

After collecting genuine evidence with necessary approvals, **copy the template outside Git into approved private storage**, populate it there, and run the same command against that private file. Even `ready_for_human_evidence_review` is **not proof** that the data, hardware timestamps, permissions or an independent calibration acquisition exist: a researcher must inspect primary evidence and separately approve any increase to independent-session n. The script *never* increments n, publishes subject/rig details, or exports input metadata. Do not store private, session-level completed manifests, personnel or patient identifiers, calibration archives, credentials or raw recordings in the public repository.
