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
    args = parser.parse_args(argv)
    if not args.audit and (args.mapping_csv or args.audit_json):
        parser.error("--mapping-csv and --audit-json require --audit")

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

    if args.preview:
        from .viz import preview
        path = preview(args.root, args.preview, camera=args.camera, subject=args.subject, repeat=args.repeat)
        print(f"Preview saved locally: {path}")


if __name__ == "__main__":
    main()
