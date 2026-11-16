## A3/D2 metadata-index sensitivity and joint-angle bias pilot

Run a *local-only* follow-up to the 26-available-participant descriptive study. This is a
**five-calibration-group anchor pilot**, not an independent validation of 26 camera
calibration sessions. It deliberately does **not** modify the previous 104 reports.

From an editable checkout, with the frozen Lite/Full assets and the existing A3 protocol
already on the Windows research PC:

```powershell
$repo = (Get-Location).Path  # run from the kinelo-sumedipose checkout
$base = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = Join-Path $repo "src"
python -m sumedipose.frame_joint_pilot --root "$base\raw" --model-dir "$base\models" --analysis-dir "$base\analysis" --output "$base\analysis\A3_D2_frame_joint_pilot_5anchors_local_20261116.json" --aggregate-output "$base\analysis\A3_D2_frame_joint_pilot_5anchors_aggregate_20261116.json"
```

- Frozen input: S1/S6/S11/S17/S23, squat A3/D2, front C1 and rear C6,
  MediaPipe Lite/Full, **24 unique RGB images per combination**, same model
  SHA256 as the earlier generalization protocol.
- Compare model 2D knee angles **only** against projected Vicon 2D angles,
  and model 3D knee angles **only** against Vicon WCS 3D included angles.
  Keep the original >=0.5 visibility filter for left/right sides.
- Report baseline angular MAE, mean signed bias and signed-error SD. These
  diagnose systematic offset but **cannot attribute it to different anatomical
  joint-centre definitions** without a separate landmark mapping.
- Evaluate fixed WCS/camera **coordinate record-index shifts**
  `(-12,-6,-3,0,3,6,12)` against the **same valid observations** at every
  shift, including shift zero. The smallest *in-sample* MAE is a sensitivity
  diagnostic only: it **does not discover real camera/Vicon exposure delay**.
  Source frame number, duplicate RGB-to-record mappings and raw hardware time
  are not interchangeable. Actual exposure synchronization remains unverified.
- Write private per-anchor reports and non-identifying aggregates **only** to
  the local Windows `D:\Research\PublicDatasets\SUMediPose\analysis`
  directory. Existing outputs cannot be overwritten. Do not upload raw JPEG,
  ZIP, joint coordinates, keypoints or participant-level JSON to GitHub,
  Second Brain, Kinelo Ops, Notion or Google Drive. S1 was involved in
  earlier ROI tuning; the pilot is not an independent holdout or a clinical
  accuracy assessment.
