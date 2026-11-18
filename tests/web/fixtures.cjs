'use strict';
// Shared synthetic data helper: a plausible 3-rep bodyweight squat as MediaPipe
// pose landmarks (normalised image + hip-centred world), so the metrics and API
// tests do not need a camera, a video file or any real person's data.

const M = require('../../web/squat-metrics.js');

function rng(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Knee angle over time: three squat cycles, 12 s at 30 fps.
function kneeAngleSeries(frames, options) {
  const opts = options || {};
  const top = opts.top === undefined ? 172 : opts.top;
  const bottom = opts.bottom === undefined ? 100 : opts.bottom;
  const reps = opts.reps === undefined ? 3 : opts.reps;
  const descent = opts.descentFrames === undefined ? 57 : opts.descentFrames;
  const ascent = opts.ascentFrames === undefined ? 36 : opts.ascentFrames;
  const hold = opts.holdFrames === undefined ? 6 : opts.holdFrames;
  const pause = opts.pauseFrames === undefined ? 21 : opts.pauseFrames;
  const cycle = descent + hold + ascent + pause;
  const out = [];
  for (let i = 0; i < frames; i++) {
    const t = i % cycle;
    let v = top;
    if (i < frames * 0.08) v = top;
    else if (t < descent) v = top - ((top - bottom) * t) / descent;
    else if (t < descent + hold) v = bottom;
    else if (t < descent + hold + ascent) v = bottom + ((top - bottom) * (t - descent - hold)) / ascent;
    else v = top;
    out.push({ i, angle: v });
  }
  // Keep only the requested number of repetitions.
  const total = reps * cycle + Math.round(frames * 0.08);
  return out.slice(0, total);
}

function squatFrames(options) {
  const opts = options || {};
  const fps = opts.fps || 30;
  const cycle =
    (opts.descentFrames === undefined ? 57 : opts.descentFrames) +
    (opts.holdFrames === undefined ? 6 : opts.holdFrames) +
    (opts.ascentFrames === undefined ? 36 : opts.ascentFrames) +
    (opts.pauseFrames === undefined ? 21 : opts.pauseFrames);
  const repsWanted = opts.reps === undefined ? 3 : opts.reps;
  const framesWanted = opts.frames || Math.max(120, repsWanted * cycle + 30);
  const series = kneeAngleSeries(framesWanted, opts);
  const noise = rng(opts.seed || 7);
  const thigh = 0.16;
  const shank = 0.16;
  const frames = [];

  for (const s of series) {
    const t = (s.i * 1000) / fps;
    const landmarks = [];
    const world = [];
    for (let i = 0; i < 33; i++) {
      landmarks.push({ x: 0.5, y: 0.5, z: 0, visibility: 0.92 });
      world.push({ x: 0, y: 0, z: 0, visibility: 0.92 });
    }
    const sides = [
      { side: 'left', sign: -1, hipX: 0.46, leftAngle: s.angle },
      { side: 'right', sign: 1, hipX: 0.54, leftAngle: s.angle - (opts.asymmetryDeg || 0) },
    ];
    for (const spec of sides) {
      const idx = M.LANDMARK_INDEX[spec.side];
      const theta = (spec.leftAngle * Math.PI) / 180;
      const alpha = 0.12 * spec.sign;
      const d = [Math.sin(alpha), Math.cos(alpha)];
      const hip = [spec.hipX, 0.44];
      const knee = [hip[0] + d[0] * thigh, hip[1] + d[1] * thigh];
      const v = [-d[0], -d[1]];
      const ankleDir = [
        Math.cos(theta) * v[0] - Math.sin(theta) * v[1],
        Math.sin(theta) * v[0] + Math.cos(theta) * v[1],
      ];
      const ankle = [knee[0] + ankleDir[0] * shank, knee[1] + ankleDir[1] * shank];
      const jitter = () => (noise() - 0.5) * (opts.jitter || 0.002);
      landmarks[idx.hip] = { x: hip[0] + jitter(), y: hip[1] + jitter(), z: 0, visibility: 0.93 };
      landmarks[idx.knee] = { x: knee[0] + jitter(), y: knee[1] + jitter(), z: 0, visibility: 0.92 };
      landmarks[idx.ankle] = { x: ankle[0] + jitter(), y: ankle[1] + jitter(), z: 0, visibility: 0.9 };
      // World landmarks: same geometry in metres, hip at the origin.
      const wTheta = theta;
      const wKnee = [d[0] * 0.42, d[1] * 0.42, 0];
      const wAnkle = [
        wKnee[0] + (Math.cos(wTheta) * v[0] - Math.sin(wTheta) * v[1]) * 0.42,
        wKnee[1] + (Math.sin(wTheta) * v[0] + Math.cos(wTheta) * v[1]) * 0.42,
        0,
      ];
      world[idx.hip] = { x: 0, y: 0, z: 0, visibility: 0.93 };
      world[idx.knee] = { x: wKnee[0], y: wKnee[1], z: 0, visibility: 0.92 };
      world[idx.ankle] = { x: wAnkle[0], y: wAnkle[1], z: 0, visibility: 0.9 };
    }
    // Shoulders above the hips, leaning a little at the bottom.
    const lean = 0.02 + (0.06 * (172 - s.angle)) / 72;
    landmarks[11] = { x: 0.44 - lean, y: 0.44 - 0.18, z: 0, visibility: 0.95 };
    landmarks[12] = { x: 0.56 - lean, y: 0.44 - 0.18, z: 0, visibility: 0.95 };
    world[11] = { x: -0.1 - lean, y: -0.32, z: 0, visibility: 0.95 };
    world[12] = { x: 0.1 - lean, y: -0.32, z: 0, visibility: 0.95 };
    frames.push(M.buildFrame(landmarks, world, opts.width || 720, opts.height || 1280, t));
  }
  return frames;
}

function metricsFixture(options) {
  return M.computeMetrics(squatFrames(options), { view: (options && options.view) || 'angle45' });
}

module.exports = { squatFrames, metricsFixture, kneeAngleSeries, rng };
