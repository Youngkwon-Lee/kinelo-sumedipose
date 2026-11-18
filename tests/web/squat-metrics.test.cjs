'use strict';
// Metrics: the included angle must match src/sumedipose/geometry.py:joint_angle
// (cross-checked in tests/test_web_metrics_parity.py), and the rep/tempo/depth
// summary must survive noise, missing landmarks and a video that stops mid-rep.

const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../../web/squat-metrics.js');
const { squatFrames, metricsFixture } = require('./fixtures.cjs');

test('included angle matches the Python definition', () => {
  assert.equal(Math.round(M.includedAngle([1, 0, 0], [0, 0, 0], [0, 1, 0])), 90);
  assert.equal(Math.round(M.includedAngle([1, 0, 0], [0, 0, 0], [-1, 0, 0])), 180);
  assert.ok(Number.isNaN(M.includedAngle([0, 0, 0], [0, 0, 0], [1, 0, 0])));
});

test('landmark indices match the research benchmark mapping', () => {
  assert.deepEqual(M.LANDMARK_INDEX.left, { shoulder: 11, hip: 23, knee: 25, ankle: 27 });
  assert.deepEqual(M.LANDMARK_INDEX.right, { shoulder: 12, hip: 24, knee: 26, ankle: 28 });
});

test('buildFrame drops a side whose landmarks are barely visible', () => {
  const landmarks = new Array(33).fill(null).map(() => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.9 }));
  landmarks[23] = { x: 0.46, y: 0.44, z: 0, visibility: 0.2 };
  landmarks[24] = { x: 0.54, y: 0.44, z: 0, visibility: 0.9 };
  landmarks[26] = { x: 0.55, y: 0.6, z: 0, visibility: 0.9 };
  landmarks[28] = { x: 0.56, y: 0.76, z: 0, visibility: 0.9 };
  const frame = M.buildFrame(landmarks, null, 720, 1280, 0);
  assert.equal(frame.left.ok, false);
  assert.ok(Number.isNaN(frame.left.knee2d));
  assert.equal(frame.right.ok, true);
  assert.ok(frame.right.knee2d > 175 && frame.right.knee2d <= 180, String(frame.right.knee2d));
});

test('three clear reps are counted with a plausible depth', () => {
  const metrics = metricsFixture({ asymmetryDeg: 7 });
  assert.equal(metrics.rep_count, 3);
  assert.ok(metrics.depth.median_deg > 90 && metrics.depth.median_deg < 110, String(metrics.depth.median_deg));
  assert.equal(metrics.depth.source, '3d');
  assert.ok(metrics.asymmetry.mean_deg >= 6 && metrics.asymmetry.mean_deg <= 8);
  assert.equal(metrics.quality.ok, true, JSON.stringify(metrics.quality.reasons));
  assert.equal(metrics.per_rep.length, 3);
  assert.match(metrics.fingerprint, /^[0-9a-f]{8}$/);
  assert.ok(metrics.flags.some((f) => f.code === 'reps'));
});

test('jitter does not split or merge repetitions', () => {
  const metrics = metricsFixture({ jitter: 0.01 });
  assert.equal(metrics.rep_count, 3);
});

test('a still video reports no repetition instead of inventing one', () => {
  const metrics = metricsFixture({ top: 170, bottom: 167 });
  assert.equal(metrics.rep_count, 0);
  assert.equal(metrics.quality.ok, false);
  assert.ok(metrics.quality.reasons.includes('no_rep_detected'));
});

test('a video that stops mid-rep does not count the unfinished rep', () => {
  const frames = squatFrames({ reps: 2 });
  const cut = frames.slice(0, frames.length - 40);
  const metrics = M.computeMetrics(cut, { view: 'angle45' });
  assert.equal(metrics.rep_count, 2);
});

test('hidden hips are reported as low visibility, not as a measurement', () => {
  const frames = squatFrames({}).map((frame) => ({
    ...frame,
    left: { ...frame.left, ok: false, knee2d: NaN, knee3d: NaN },
    right: { ...frame.right, ok: false, knee2d: NaN, knee3d: NaN },
    mean2d: NaN,
    mean3d: NaN,
  }));
  const metrics = M.computeMetrics(frames, { view: 'angle45' });
  assert.equal(metrics.quality.ok, false);
  assert.ok(metrics.quality.reasons.includes('low_visibility'));
});

test('view label is carried into the metrics', () => {
  assert.equal(metricsFixture({ view: 'front' }).view_label, '정면');
  assert.equal(M.cameraViewLabel('side'), '측면');
  assert.equal(M.cameraViewLabel(undefined), '미지정');
});

test('fingerprint changes when the measurement changes', () => {
  const a = metricsFixture({ asymmetryDeg: 3 });
  const b = metricsFixture({ asymmetryDeg: 9 });
  assert.notEqual(a.fingerprint, b.fingerprint);
});
