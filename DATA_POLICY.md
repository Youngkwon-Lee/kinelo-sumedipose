# Data governance and publication rules

This is a **public code-only** research repository.

## Do not commit

- SUMediPose RGB pictures, videos, WCS or camera 3D/2D coordinate records, per-participant metadata and subject CSVs.
- Raw ZIP archives, original calibration files, internal data, Dataverse manifests, file inventories, or MD5 logs.
- Derived previews, movement plots showing identifiable people, feature matrices, subject-level outputs, model predictions, or any personally identifying information.
- Credentials, access tokens, machine logs, private research notes or absolute local folder listings.

The repository's `.gitignore` is a safety layer, **not** permission to publish an otherwise sensitive file. Always inspect staged changes using `git diff --cached --name-only` and `git diff --cached`.

## Source and data terms

Get SUMediPose **directly from the official distributor**. Review the applicable dataset license, permitted uses, citation requirements, participant privacy limitations, and your institutional research approval before using or sharing it. No rights to redistribute SUMediPose originals are implied by this repository's MIT license.

## Research limitations

This package is an exploratory research aid, not a validated clinical interpretation system. Coordinate units, temporal alignment, subject-level grouping and data provenance must be verified before calculating publishable metrics. Do not treat one-frame, first-record comparisons as synchronized proof.

## Known local example (not part of Git)

On the authorized workstation, originals were kept separately at `D:\Research\PublicDatasets\SUMediPose\raw`. This path is a local usage example, not an upload target. It is not needed to run CI.
