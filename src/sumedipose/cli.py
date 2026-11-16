"""Small local-only command-line interface for SUMediPose A3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry import knee_angles
from .io import read_sample, record_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect local SUMediPose original ZIP files")
    parser.add_argument("--root", required=True, help="Local SUMediPose/raw directory")
    parser.add_argument("--camera", default="C1")
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--repeat", default="D1")
    parser.add_argument("--preview", help="Optional output PNG path (requires .[viz])")
    parser.add_argument("--audit", action="store_true", help="Audit A3 WCS-camera-RGB record mappings")
    parser.add_argument("--mapping-csv", help="Local output path for per-record mapping (requires --audit)")
    parser.add_argument("--audit-json", help="Local output path for aggregate audit (requires --audit)")
    parser.add_argument("--motion-audit", action="store_true", help="Compare RGB optical flow with projected torso motion")
    parser.add_argument("--motion-json", help="Local-only aggregate motion audit JSON (requires --motion-audit)")
    parser.add_argument("--stride", type=int, default=6, help="Unique RGB frames per optical-flow pair")
    parser.add_argument("--image-width", type=int, default=288, help="Resized image width in pixels")
    parser.add_argument("--motion-axis", choices=("x", "y"), default="y")
    parser.add_argument("--roi-markers", nargs="+", default=["C7", "CLAV"], help="ROI center markers")
    parser.add_argument("--roi-half-width", type=int, default=32)
    parser.add_argument("--roi-half-height", type=int, default=36)
    parser.add_argument("--wand-fit", action="store_true", help="Compare source LED annotations with wand_xy, not independent accuracy")
    parser.add_argument("--wand-json", help="Local-only source wand residual JSON")
    parser.add_argument("--timestamp-evidence", action="store_true", help="Audit availability of raw capture time fields")
    parser.add_argument("--timestamp-json", help="Local-only timestamp evidence inventory JSON")
    parser.add_argument("--calibration-inventory", action="store_true", help="Inspect source-provided calibration matrix structure")
    parser.add_argument("--calibration-json", help="Local calibration inventory JSON (requires --calibration-inventory)")
    args = parser.parse_args(argv)
    if not args.audit and (args.mapping_csv or args.audit_json):
        parser.error("--mapping-csv and --audit-json require --audit")

    if args.motion_json and not args.motion_audit:
        parser.error("--motion-json requires --motion-audit")
    if args.calibration_json and not args.calibration_inventory:
        parser.error("--calibration-json requires --calibration-inventory")
    if args.wand_json and not args.wand_fit:
        parser.error("--wand-json requires --wand-fit")
    if args.timestamp_json and not args.timestamp_evidence:
        parser.error("--timestamp-json requires --timestamp-evidence")

    if args.audit:
        from .sync import audit_sample, save_mapping
        report, rows = audit_sample(args.root, args.camera, args.subject, args.repeat)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.mapping_csv:
            path = save_mapping(rows, args.mapping_csv)
            print(f"Mapping saved locally: {path}")
        if args.audit_json:
            path = Path(args.audit_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Audit saved locally: {path}")
    else:
        _, world, camera = read_sample(args.root, camera=args.camera, subject=args.subject, repeat=args.repeat)
        summary = record_summary(world, camera)
        if world:
            summary["first_wcs_knee_angles"] = knee_angles(world[0])
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    def save_json(path, record):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    if args.motion_audit:
        from .temporal import optical_flow_audit
        finding = optical_flow_audit(
            args.root, args.camera, args.subject, args.repeat, stride=args.stride,
            image_width=args.image_width, marker_ids=tuple(args.roi_markers),
            displacement_axis=args.motion_axis,
            roi_half_width=args.roi_half_width, roi_half_height=args.roi_half_height,
        )
        print(json.dumps(finding, ensure_ascii=False, indent=2))
        if args.motion_json:
            print(f"Motion report saved locally: {save_json(args.motion_json, finding)}")

    if args.calibration_inventory:
        from .calibration import external_calibration_inventory
        finding = external_calibration_inventory(args.root, args.subject)
        print(json.dumps(finding, ensure_ascii=False, indent=2))
        if args.calibration_json:
            print(f"Calibration report saved locally: {save_json(args.calibration_json, finding)}")

    if args.wand_fit:
        from .calibration import calibration_wand_fit_residuals
        finding = calibration_wand_fit_residuals(args.root, args.subject, repeat=args.repeat)
        print(json.dumps(finding, ensure_ascii=False, indent=2))
        if args.wand_json:
            print(f"Wand fit summary saved locally: {save_json(args.wand_json, finding)}")

    if args.timestamp_evidence:
        from .sync_evidence import timestamp_evidence_inventory
        finding = timestamp_evidence_inventory(args.root, args.camera, args.subject, args.repeat)
        print(json.dumps(finding, ensure_ascii=False, indent=2))
        if args.timestamp_json:
            print(f"Timestamp evidence saved locally: {save_json(args.timestamp_json, finding)}")

    if args.preview:
        from .viz import preview
        path = preview(args.root, args.preview, camera=args.camera, subject=args.subject, repeat=args.repeat)
        print(f"Preview saved locally: {path}")


if __name__ == "__main__":
    main()
