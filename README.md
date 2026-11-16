# Kinelo × SUMediPose

Open-source **code-only** toolkit for exploratory analysis of the [SUMediPose](https://doi.org/10.1016/j.dib.2025.111579) dataset, with an initial focus on **A3 (squat)** RGB images, world-coordinate (WCS) 3D markers, and camera-coordinate 2D/3D markers.

> **Research status:** Early prototype. Not a medical device, clinical decision tool, or validated motion-assessment system. No participant images, motion data, or original dataset archives are included in this repository.

## What this does

- Read existing SUMediPose ZIP archives in-place; **never extract or modify raw data**.
- Identify A3 squat samples by camera, subject, and repeat (for example, `C1/S1/A3/D1`).
- Summarize records, frame identifiers, and marker IDs.
- Calculate preliminary hip–knee–ankle included angles from WCS 3D coordinates, in degrees.
- Optionally save a **local** three-panel PNG showing one RGB frame, camera marker overlays, and exploratory WCS/camera 3D marker clouds.
- Run dependency-light tests without downloading the dataset.

## Local setup

Requires Python 3.10+:

```powershell
git clone https://github.com/Youngkwon-Lee/kinelo-sumedipose.git
cd kinelo-sumedipose
python -m pip install -e ".[test,viz]"
python -m pytest -q
```

Data must be acquired separately from the official SUMediPose source, following its own terms. This repo assumes a dataset tree like:

```text
D:\Research\PublicDatasets\SUMediPose\raw\
  frames\C1\S1\C1S1A3D1.zip
  WCS\S1.zip
  internal_data\C1\S1.zip
```

Inspect and calculate example knee angles:

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw"
```

Generate a **local, non-publishable** preview (the image contains participant imagery):

```powershell
sumedipose --root "D:\Research\PublicDatasets\SUMediPose\raw" --preview "D:\Research\PublicDatasets\SUMediPose\analysis\C1S1A3D1_preview.png"
```

The image is intentionally created **outside the Git repository**. Do not publish it without assessing dataset terms, informed consent and ethics requirements.

## Important interpretation caveats

1. WCS and camera record lists can both contain 1,205 items while the RGB sample has 928 files; `frame_num` values are **not necessarily identical**. Index-aligned WCS/camera records in the prototype are only an exploratory preview, **not validated temporal synchronization**.
2. WCS and camera 3D coordinates appear to use different scales. **Do not assume identical length units** or use absolute distances across systems without checking the data documentation and calibration.
3. Marker IDs such as `LHJC`, `LKJC`, `LAJC` and their right-side counterparts are used for example included knee angles, **not clinical knee flexion** or independently validated biomechanics.
4. Annotations and calibration can differ across subjects and recordings. Benchmarking requires frame alignment, projection validation, coordinate conventions, quality control, and independent evaluation.

## Roadmap

- [x] Code-only repository and strict data exclusion.
- [x] Original ZIP sample reader, preliminary geometry and preview.
- [ ] Explicit validated frame synchronization and dropped-frame handling.
- [ ] Documented calibration and coordinate conversions.
- [ ] Verified joint kinematics with anatomical conventions.
- [ ] Benchmark adapters for MeTRAbs / GVHMR / other pose models.
- [ ] Per-subject evaluation, uncertainty and bias reporting (subject-wise splits).

## Privacy and dataset

See [DATA_POLICY.md](DATA_POLICY.md). Research originals must remain on local storage, and neither the archives, per-subject metadata nor preview images belong in Git history.

## License

Our **original code and documentation only**: MIT; see [LICENSE](LICENSE). This does **not** grant rights over SUMediPose data or third-party software.
