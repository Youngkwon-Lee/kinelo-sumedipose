"""The browser metrics must agree with the Python package they were ported from.

Two things are checked here, both against the real code rather than a copy:

1. `web/squat-metrics.js:includedAngle` returns the same number as
   `sumedipose.geometry.joint_angle` for random inputs (1e-6 degrees).
2. The MediaPipe landmark indices the browser uses are the ones
   `sumedipose.benchmark` maps (`hip 23/25/27`, `knee 24/26/28`), so the two
   pipelines measure the same anatomical points.

The JavaScript unit tests (`tests/web/*.test.cjs`) and the API tests are run
from here too, so the existing Python CI exercises the web code without any
workflow change. They are skipped when Node is not installed.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from sumedipose import benchmark
from sumedipose.geometry import joint_angle

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "web" / "parity.cjs"


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed; web metrics tests skipped")
    try:
        major = int(subprocess.run([node, "-v"], capture_output=True, text=True, timeout=20).stdout.strip().lstrip("v").split(".")[0])
    except (ValueError, OSError, subprocess.SubprocessError):  # pragma: no cover - exotic environments
        pytest.skip("Node.js version could not be read")
    if major < 18:
        pytest.skip(f"Node.js {major} is too old for `node --test`")
    return node


def _run_js(triples):
    node = _node()
    proc = subprocess.run(
        [node, str(PARITY)],
        input=json.dumps(triples),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_web_geometry_matches_python_joint_angle():
    rnd = random.Random(20261118)
    triples = []
    for _ in range(200):
        triples.append([[rnd.uniform(-2, 2) for _ in range(3)] for _ in range(3)])
    # Degenerate and straight cases must agree too, including NaN.
    triples.append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    triples.append([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    triples.append([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = _run_js(triples)
    js_angles = result["angles"]
    assert len(js_angles) == len(triples)
    for triple, js_angle in zip(triples, js_angles):
        py_angle = joint_angle(*triple)
        if math.isnan(py_angle):
            assert js_angle is None or (isinstance(js_angle, float) and math.isnan(js_angle))
            continue
        assert math.isclose(py_angle, js_angle, rel_tol=0, abs_tol=1e-6), (triple, py_angle, js_angle)


def test_web_landmark_indices_match_benchmark_mapping():
    result = _run_js([[[1, 0, 0], [0, 0, 0], [0, 1, 0]]])
    indices = result["indices"]
    for side in ("left", "right"):
        reference, predicted = benchmark.JOINT_NAMES[side]["reference"], benchmark.JOINT_NAMES[side]["pred"]
        assert predicted == (indices[side]["hip"], indices[side]["knee"], indices[side]["ankle"]), side
        assert reference[0][0] == ("L" if side == "left" else "R")


def test_web_suite_passes():
    """Runs the browser-metrics, catalog, API and view tests through Node."""
    node = _node()
    proc = subprocess.run(
        [node, "--test", "tests/web/*.test.cjs"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
        shell=False,
    )
    if proc.returncode != 0 and "Could not find" in (proc.stderr + proc.stdout):
        # Older Node without glob support: list the files explicitly.
        files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "web").glob("*.test.cjs"))
        proc = subprocess.run(
            [node, "--test", *files],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(ROOT),
        )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
