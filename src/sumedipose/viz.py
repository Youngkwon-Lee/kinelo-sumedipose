"""Static research preview, deliberately not claiming validated time synchronization."""
from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def preview(root: str | Path, output: str | Path, camera="C1", subject="S1", repeat="D1") -> Path:
    """Plot first index-aligned WCS and camera records, and its referenced RGB image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install optional visualization dependencies: pip install -e '.[viz]'") from exc

    from .io import read_sample
    paths, world, camera_records = read_sample(root, camera=camera, subject=subject, repeat=repeat)
    if not world or not camera_records:
        raise ValueError("Sample records are empty")
    wc, cam = world[0], camera_records[0]
    name = Path(cam["path"]).name
    with ZipFile(paths["rgb"]) as archive:
        image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(131)
    ax.imshow(image)
    ids, xy = cam["point_ids"], np.asarray(cam["xy"], dtype=float)
    for joint in ("LASI", "RASI", "LHJC", "RHJC", "LKJC", "RKJC", "LAJC", "RAJC", "C7", "T10"):
        if joint in ids:
            x, y = xy[ids.index(joint)]
            if np.isfinite([x, y]).all():
                ax.scatter([x], [y], s=18)
                ax.annotate(joint, (x, y), color="yellow", fontsize=7)
    ax.set_title(f"RGB {name} / camera markers")
    ax.axis("off")

    for index, record, title in ((132, wc, "World-coordinate markers"), (133, cam, "Camera-coordinate markers")):
        panel = fig.add_subplot(index, projection="3d")
        xyz = np.asarray(record["xyz"], dtype=float)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.size:
            panel.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], s=5)
        panel.set_xlabel("X")
        panel.set_ylabel("Z")
        panel.set_zlabel("Y")
        panel.set_title(title)

    fig.suptitle("Exploratory preview only: first record indices, timing not validated")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output
