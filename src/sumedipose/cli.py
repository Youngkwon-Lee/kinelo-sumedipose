"""Small local-only command-line interface for SUMediPose A3."""
from __future__ import annotations

import argparse
import json

from .geometry import knee_angles
from .io import read_sample, record_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect local SUMediPose original ZIP files")
    parser.add_argument("--root", required=True, help="Local SUMediPose/raw directory")
    parser.add_argument("--camera", default="C1")
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--repeat", default="D1")
    parser.add_argument("--preview", help="Optional output PNG path (requires .[viz])")
    args = parser.parse_args(argv)
    paths, world, camera = read_sample(args.root, camera=args.camera, subject=args.subject, repeat=args.repeat)
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
