# SUMediPose A3/D2: camera-plane vs 3D knee-angle decomposition (exploratory)

This is a **code-only, local-data** experiment. It tests the geometric identity

```text
(model 3D angle - Vicon camera 3D angle)
  = (model XY-flattened angle - Vicon camera XY-flattened angle)
  + [(model 3D - model XY) - (Vicon camera 3D - Vicon camera XY)]
```

Both 3D triplets are expressed in their **own** coordinate systems:
MediaPipe world landmarks (hip/knee/ankle) and source Vicon camera `xyz`
joint centres. Vicon WCS included angles are used as a consistency readback
(required to differ by <= 1 degree), not as an assumption of physical
landmark alignment. The decomposition is exact **for included angles** and
does not require 3D axis-to-axis registration. It **does require caution**
when interpreting the flattened camera/model XY terms: the source camera
coordinates and the model world coordinates have different origins,
landmark definitions and potentially different axis conventions.

The last bracket is an **out-of-plane angular-effect mismatch** (a geometric
axis-dependent *term*), **not independently validated Z-coordinate error**.
Further analysis of the dataset's Vicon WCS versus derived camera XYZ found
an **axis-parity reversal**, so this model-camera XY decomposition must not
be described as physical depth contribution without independent model-to-
reference axis registration. Anatomical
joint-centre definitions, camera extrinsics, model priors, image calibration,
perspective projection and unverified exposure timing can contribute.
Do not interpret `abs_flat_xy_error` as 2D landmark precision or the
depth term as a validated physical depth-measurement error. The separate
`abs_image2d_error` compares MediaPipe image landmarks against Vicon
projected image angles and is reported in its own reference space.

Model and sample selection follow the previously frozen 26-participant A3 D2
protocol: 26 RGB-available subjects (S5/S10 excluded), C1/C6 × Lite/Full,
24 unique RGB images per recording, visibility >=0.5, exact original model
asset SHA256. **Use unique source-valid `dup_num == factor` Vicon records**, not
the historical midpoint-index association. Each sample retains exactly the
same selected RGB filename and order as the original pilot. All **104 prior
source-valid paired-recheck reports** in `strict_source_pairing_20261117/`
must reproduce their side-specific valid counts, detections and 2D/3D MAE
before the new decomposition is released. The old midpoint baselines are
retained unchanged and must not be overwritten. The preliminary `anchors`
option is a five-group diagnostic; it is not an independent holdout.

Subjects are nested within five existing calibration groups. Aggregate
each subject's valid side angles first, then equally average subject means
within group, then equally average the five group means. Angle bands
(reference Vicon included angle under 120°, 120–150°, >=150°) are
**descriptive, not clinical flexion thresholds**. The per-band group
coverage must be reported; no p-values, causal statements or clinical
cutoffs are inferred.

Run from the checkout on the approved Windows research PC:

```powershell
$base = "D:\Research\PublicDatasets\SUMediPose"
$env:PYTHONPATH = (Join-Path (Get-Location).Path "src")
python -m sumedipose.depth_angle_decomposition --root "$base\raw" --model-dir "$base\models" --analysis-dir "$base\analysis" --scope available26 --private-output "$base\analysis\A3_D2_depth_decomposition_26_sourcevalid_v2_local_20261117.json" --aggregate-output "$base\analysis\A3_D2_depth_decomposition_26_sourcevalid_v2_aggregate_20261117.json"
```

Outputs exist **only on Windows D:**, are created without overwriting, and
contain no original RGBs or per-frame AI landmarks. The private output
contains per-participant summaries; the aggregate contains no participant IDs.
Neither output JSON should be committed to Git. Raw datasets, source camera
records, model weights and any individual-level reports stay local. Only
reviewed non-identifying aggregate interpretation may be captured later in
the existing Second Brain research Candidate.

**Interpretation gate:** Even if the out-of-plane angular term dominates,
a working depth correction algorithm has **not** been learned or validated.
Next measure a genuinely matched anatomical landmark definition, check
camera–Vicon capture timestamp evidence and validate on a separate
calibration date/device before estimating generalization or clinical efficacy.
