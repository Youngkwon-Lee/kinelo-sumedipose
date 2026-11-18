/* Squat movement metrics — browser + Node, no DOM, no network.
 *
 * The included-angle maths is a direct port of `src/sumedipose/geometry.py:
 * joint_angle`, and the landmark indices are the same ones `src/sumedipose/
 * benchmark.py` uses for the MediaPipe pose baseline (hip 23/24, knee 25/26,
 * ankle 27/28). `tests/test_web_metrics_parity.py` checks both against the
 * Python package, so the numbers a buyer sees come from the same definition as
 * the research code — not a lookalike.
 *
 * What this module does NOT claim: it is not a clinical measure. A single
 * camera gives a projected 2D angle in the image plane and a monocular 3D
 * estimate from the pose model's hip-centred world landmarks. Neither is
 * Vicon-registered or exposure-time synchronised, exactly as the repository's
 * README states for its own benchmarks. The UI and the paid report repeat that.
 */
(function (global) {
  'use strict';

  const LANDMARK_INDEX = {
    left: { shoulder: 11, hip: 23, knee: 25, ankle: 27 },
    right: { shoulder: 12, hip: 24, knee: 26, ankle: 28 },
  };

  const MIN_VISIBILITY = 0.5;
  const SIDES = ['left', 'right'];
  const VERSION = 1;

  function cameraViewLabel(view) {
    if (view === 'front') return '정면';
    if (view === 'side') return '측면';
    if (view === 'angle45') return '45도 사선';
    return '미지정';
  }

  /* Same definition as geometry.joint_angle: included angle at `b` in degrees,
   * NaN for degenerate vectors. Pass z=0 for the image-plane angle. */
  function includedAngle(a, b, c) {
    const ax = a[0] - b[0], ay = a[1] - b[1], az = (a[2] || 0) - (b[2] || 0);
    const cx = c[0] - b[0], cy = c[1] - b[1], cz = (c[2] || 0) - (b[2] || 0);
    const na = Math.sqrt(ax * ax + ay * ay + az * az);
    const nc = Math.sqrt(cx * cx + cy * cy + cz * cz);
    const norm = na * nc;
    if (!Number.isFinite(norm) || norm <= 1e-12) return NaN;
    const cos = (ax * cx + ay * cy + az * cz) / norm;
    return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
  }

  function point2d(p, width, height) {
    return [p.x * width, p.y * height, 0];
  }

  function point3d(p) {
    return [p.x, p.y, p.z];
  }

  function visibilityOf(points) {
    let v = 1;
    for (const p of points) {
      const value = typeof p.visibility === 'number' ? p.visibility : 1;
      v = Math.min(v, value);
    }
    return v;
  }

  function round(value, digits) {
    if (!Number.isFinite(value)) return NaN;
    const f = Math.pow(10, digits === undefined ? 1 : digits);
    return Math.round(value * f) / f;
  }

  /* One video frame -> per-side angles plus the couple of global angles the
   * report talks about. `landmarks` are MediaPipe pose landmarks in normalised
   * image coordinates; `worldLandmarks` are the same model's world landmarks. */
  function buildFrame(landmarks, worldLandmarks, width, height, timestampMs) {
    const frame = {
      t: Number(timestampMs) || 0,
      left: { ok: false, knee2d: NaN, knee3d: NaN, visibility: 0 },
      right: { ok: false, knee2d: NaN, knee3d: NaN, visibility: 0 },
      mean2d: NaN,
      mean3d: NaN,
      trunk2d: NaN,
      stance: NaN,
    };
    if (!Array.isArray(landmarks)) return frame;
    const world = Array.isArray(worldLandmarks) ? worldLandmarks : null;

    for (const side of SIDES) {
      const idx = LANDMARK_INDEX[side];
      const hip = landmarks[idx.hip];
      const knee = landmarks[idx.knee];
      const ankle = landmarks[idx.ankle];
      if (!hip || !knee || !ankle) continue;
      const vis = visibilityOf([hip, knee, ankle]);
      frame[side].visibility = round(vis, 2);
      if (vis < MIN_VISIBILITY) continue;
      frame[side].ok = true;
      frame[side].knee2d = includedAngle(
        point2d(hip, width, height),
        point2d(knee, width, height),
        point2d(ankle, width, height)
      );
      if (world) {
        const wHip = world[idx.hip], wKnee = world[idx.knee], wAnkle = world[idx.ankle];
        if (wHip && wKnee && wAnkle) {
          frame[side].knee3d = includedAngle(point3d(wHip), point3d(wKnee), point3d(wAnkle));
        }
      }
    }

    const ok2d = SIDES.map((s) => frame[s].knee2d).filter(Number.isFinite);
    const ok3d = SIDES.map((s) => frame[s].knee3d).filter(Number.isFinite);
    if (ok2d.length) frame.mean2d = ok2d.reduce((a, b) => a + b, 0) / ok2d.length;
    if (ok3d.length) frame.mean3d = ok3d.reduce((a, b) => a + b, 0) / ok3d.length;

    const shoulders = [landmarks[LANDMARK_INDEX.left.shoulder], landmarks[LANDMARK_INDEX.right.shoulder]];
    const hips = [landmarks[LANDMARK_INDEX.left.hip], landmarks[LANDMARK_INDEX.right.hip]];
    if (shoulders.every(Boolean) && hips.every(Boolean) && visibilityOf(shoulders.concat(hips)) >= MIN_VISIBILITY) {
      const shoulderMid = [
        ((shoulders[0].x + shoulders[1].x) / 2) * width,
        ((shoulders[0].y + shoulders[1].y) / 2) * height,
      ];
      const hipMid = [
        ((hips[0].x + hips[1].x) / 2) * width,
        ((hips[0].y + hips[1].y) / 2) * height,
      ];
      const dx = Math.abs(shoulderMid[0] - hipMid[0]);
      const dy = Math.abs(shoulderMid[1] - hipMid[1]);
      frame.trunk2d = (Math.atan2(dx, dy) * 180) / Math.PI;
      const shoulderWidth = Math.abs(shoulders[0].x - shoulders[1].x) * width;
      const ankleGap = Math.abs(landmarks[LANDMARK_INDEX.left.ankle] ? landmarks[LANDMARK_INDEX.left.ankle].x - (landmarks[LANDMARK_INDEX.right.ankle] || {}).x : 0) * width;
      if (shoulderWidth > 1 && Number.isFinite(ankleGap)) frame.stance = ankleGap / shoulderWidth;
    }
    return frame;
  }

  function percentile(values, p) {
    const arr = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
    if (!arr.length) return NaN;
    const at = (arr.length - 1) * p;
    const lo = Math.floor(at);
    const hi = Math.ceil(at);
    if (lo === hi) return arr[lo];
    return arr[lo] + (arr[hi] - arr[lo]) * (at - lo);
  }

  function median(values) {
    return percentile(values, 0.5);
  }

  function sd(values) {
    const arr = values.filter(Number.isFinite);
    if (arr.length < 2) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (arr.length - 1);
    return Math.sqrt(variance);
  }

  /* Bottom-of-squat detection on the knee-angle signal with hysteresis, so a
   * noisy frame does not split one repetition into two. Angles shrink as the
   * person descends (180 = straight leg). */
  function detectRepRuns(signal, options) {
    const opts = options || {};
    const minRepMs = opts.minRepMs === undefined ? 350 : opts.minRepMs;
    const mergeGapMs = opts.mergeGapMs === undefined ? 320 : opts.mergeGapMs;
    const values = signal.map((s) => s.v).filter(Number.isFinite);
    if (values.length < 6) return { runs: [], top: NaN, bottom: NaN, enter: NaN };

    const top = percentile(values, 0.95);
    const bottom = percentile(values, 0.05);
    const range = top - bottom;
    if (!Number.isFinite(range) || range < 20) return { runs: [], top, bottom, enter: NaN, range };
    const enter = bottom + range * 0.55;
    const exit = bottom + range * 0.78;

    const runs = [];
    let cur = null;
    signal.forEach((s, position) => {
      if (!Number.isFinite(s.v)) return;
      if (!cur) {
        if (s.v < enter) cur = { start: s, end: s, min: s, startIdx: position, endIdx: position };
        return;
      }
      if (s.v < cur.min.v) cur.min = s;
      if (s.v > exit) {
        cur.end = s;
        cur.endIdx = position;
        cur.closed = true;
        runs.push(cur);
        cur = null;
      } else {
        cur.end = s;
        cur.endIdx = position;
      }
    });
    if (cur) runs.push(cur);

    // The hysteresis window starts a rep only once the knees are already bent;
    // widen each rep back out to the last/first standing frame so descent and
    // ascent times cover the whole movement.
    for (const run of runs) {
      let i = run.startIdx;
      while (i > 0 && Number.isFinite(signal[i - 1].v) && signal[i - 1].v < exit) i -= 1;
      run.start = signal[i];
      run.startIdx = i;
      let j = run.endIdx;
      while (j < signal.length - 1 && Number.isFinite(signal[j + 1].v) && signal[j + 1].v < exit) j += 1;
      run.end = signal[j];
      run.endIdx = j;
    }

    const merged = [];
    for (const run of runs) {
      const last = merged[merged.length - 1];
      if (last && run.start.t - last.end.t <= mergeGapMs) {
        last.end = run.end;
        last.endIdx = run.endIdx;
        last.closed = run.closed === true;
        if (run.min.v < last.min.v) last.min = run.min;
      } else {
        merged.push({
          start: run.start,
          end: run.end,
          min: run.min,
          startIdx: run.startIdx,
          endIdx: run.endIdx,
          closed: run.closed === true,
        });
      }
    }
    const kept = merged.filter(
      (r) => r.closed === true && r.end.t - r.start.t >= minRepMs && r.min.v < enter
    );
    return { runs: kept, top, bottom, enter, exit, range };
  }

  function symmetryLevel(degrees) {
    if (!Number.isFinite(degrees)) return 'unknown';
    if (degrees < 4) return 'small';
    if (degrees < 8) return 'moderate';
    return 'large';
  }

  function depthLevel(degrees3d) {
    if (!Number.isFinite(degrees3d)) return 'unknown';
    if (degrees3d >= 125) return 'shallow';
    if (degrees3d >= 105) return 'moderate';
    if (degrees3d >= 80) return 'deep';
    return 'very_deep';
  }

  /* Main entry point: a series of frames -> the numeric summary that both the
   * free result screen and the paid report API receive. */
  function computeMetrics(frames, options) {
    const opts = Object.assign({ view: 'angle45', minRepMs: 350, mergeGapMs: 320 }, options || {});
    const all = (Array.isArray(frames) ? frames : []).filter((f) => f && Number.isFinite(f.t));
    const usable = all.filter((f) => Number.isFinite(f.mean2d) || Number.isFinite(f.mean3d));
    const use3d =
      usable.length > 0 &&
      usable.filter((f) => Number.isFinite(f.mean3d)).length / usable.length >= 0.6;
    const key = use3d ? 'mean3d' : 'mean2d';
    const signal = usable
      .filter((f) => Number.isFinite(f[key]))
      .map((f, i) => ({ i, t: f.t, v: f[key], frame: f }));

    const detection = detectRepRuns(signal, opts);
    const perRep = [];
    detection.runs.forEach((run, index) => {
      const bottomFrame = run.min.frame;
      const breakDown = { index: index + 1, startMs: run.start.t, bottomMs: run.min.t, endMs: run.end.t };
      for (const side of SIDES) {
        breakDown[side + 'Knee2d'] = round(bottomFrame[side].knee2d, 1);
        breakDown[side + 'Knee3d'] = round(bottomFrame[side].knee3d, 1);
      }
      const asymBasis = use3d
        ? Math.abs(bottomFrame.left.knee3d - bottomFrame.right.knee3d)
        : Math.abs(bottomFrame.left.knee2d - bottomFrame.right.knee2d);
      breakDown.minKnee = round(run.min.v, 1);
      breakDown.asymmetry = round(asymBasis, 1);
      breakDown.descentS = round((run.min.t - run.start.t) / 1000, 2);
      breakDown.ascentS = round((run.end.t - run.min.t) / 1000, 2);
      breakDown.durationS = round((run.end.t - run.start.t) / 1000, 2);
      breakDown.trunk2d = round(bottomFrame.trunk2d, 1);
      breakDown.stance = round(bottomFrame.stance, 2);
      perRep.push(breakDown);
    });

    const depthValues = perRep.map((r) => r.minKnee).filter(Number.isFinite);
    const asymValues = perRep.map((r) => r.asymmetry).filter(Number.isFinite);
    const durationValues = perRep.map((r) => r.durationS).filter(Number.isFinite);
    const descent = perRep.map((r) => r.descentS).filter(Number.isFinite);
    const ascent = perRep.map((r) => r.ascentS).filter(Number.isFinite);
    const trunk = perRep.map((r) => r.trunk2d).filter(Number.isFinite);
    const stance = perRep.map((r) => r.stance).filter(Number.isFinite);
    const depth2dValues = perRep
      .map((r) => {
        const sides = [r.leftKnee2d, r.rightKnee2d].filter(Number.isFinite);
        return sides.length ? sides.reduce((a, b) => a + b, 0) / sides.length : NaN;
      })
      .filter(Number.isFinite);

    const durationS = all.length > 1 ? (all[all.length - 1].t - all[0].t) / 1000 : 0;
    const validRatio = all.length ? usable.length / all.length : 0;
    const frameCount = all.length;

    const reasons = [];
    if (frameCount < 30) reasons.push('too_few_frames');
    if (validRatio < 0.6) reasons.push('low_visibility');
    if (!perRep.length) reasons.push('no_rep_detected');
    else if (detection.range < 25) reasons.push('small_range');
    if (durationS > 0 && durationS < 4) reasons.push('too_short');

    const metrics = {
      version: VERSION,
      view: opts.view,
      view_label: cameraViewLabel(opts.view),
      angle_source: use3d ? 'world_3d' : 'image_2d',
      frames: frameCount,
      frames_used: usable.length,
      valid_ratio: round(validRatio, 3),
      duration_s: round(durationS, 1),
      rep_count: perRep.length,
      range_deg: round(detection.range, 1),
      depth: {
        level: depthLevel(median(depthValues)),
        median_deg: round(use3d ? median(depthValues) : median(depth2dValues), 1),
        min_deg: round(Math.min.apply(null, depthValues.length ? depthValues : [NaN]), 1),
        sd_deg: round(sd(depthValues), 1),
        source: use3d ? '3d' : '2d',
      },
      depth_2d_median_deg: round(median(depth2dValues), 1),
      asymmetry: {
        level: symmetryLevel(median(asymValues)),
        mean_deg: round(
          asymValues.length ? asymValues.reduce((a, b) => a + b, 0) / asymValues.length : NaN,
          1
        ),
        max_deg: round(Math.max.apply(null, asymValues.length ? asymValues : [NaN]), 1),
      },
      tempo: {
        rep_median_s: round(median(durationValues), 2),
        descent_median_s: round(median(descent), 2),
        ascent_median_s: round(median(ascent), 2),
        rhythm: round(median(descent) / (median(ascent) || NaN), 2),
        sd_s: round(sd(durationValues), 2),
      },
      trunk: { lean_median_deg: round(median(trunk), 1), lean_max_deg: round(Math.max.apply(null, trunk.length ? trunk : [NaN]), 1) },
      stance: { ratio_median: round(median(stance), 2) },
      consistency: { depth_sd_deg: round(sd(depthValues), 1), tempo_sd_s: round(sd(durationValues), 2) },
      per_rep: perRep.slice(0, 40),
      quality: { ok: reasons.length === 0, reasons },
      measurement_notes: [
        '각도는 한 대의 카메라 영상에서 계산한 값입니다. 2D 각도는 화면 평면에 투영된 각도이고, 3D 각도는 포즈 모델이 추정한 골반 중심 좌표에서 계산한 값입니다.',
        '동작 시간은 무릎 각도가 서 있는 기준선을 지나는 순간을 기준으로 나눈 값입니다.',
        '광학 모션캡처(Vicon 등)로 검증된 값이 아니며, 카메라 각도·거리·옷·가림에 따라 달라집니다.',
        '같은 자리·같은 각도에서 다시 재면 그 변화 비교에는 쓸 수 있습니다.',
      ],
    };
    metrics.flags = describeMetrics(metrics);
    metrics.fingerprint = fingerprint(metrics);
    return metrics;
  }

  /* Plain-language observations. Deliberately no judgement words: the report
   * model gets to interpret, and the disclaimers stay with the app. */
  function describeMetrics(metrics) {
    const out = [];
    if (!metrics) return out;
    const m = metrics;
    if (m.rep_count) out.push({ code: 'reps', text: '스쿼트 ' + m.rep_count + '회를 인식했습니다.' });
    if (Number.isFinite(m.depth.median_deg)) {
      const label = { shallow: '얕은 편', moderate: '중간 깊이', deep: '깊은 편', very_deep: '매우 깊은 편' }[m.depth.level] || '';
      out.push({ code: 'depth', text: '가장 깊은 지점의 무릎 각도는 중간값 ' + m.depth.median_deg + '도(' + label + ')입니다.' });
    }
    if (Number.isFinite(m.asymmetry.mean_deg)) {
      const label = { small: '작음', moderate: '중간', large: '큼', unknown: '판단 보류' }[m.asymmetry.level] || '';
      out.push({ code: 'asymmetry', text: '좌우 무릎 각도 차이는 평균 ' + m.asymmetry.mean_deg + '도(' + label + ')입니다.' });
    }
    if (Number.isFinite(m.tempo.rep_median_s)) {
      out.push({
        code: 'tempo',
        text: '한 번에 ' + m.tempo.rep_median_s + '초, 내려가는 구간 ' + m.tempo.descent_median_s + '초 / 올라오는 구간 ' + m.tempo.ascent_median_s + '초입니다.',
      });
    }
    if (Number.isFinite(m.trunk.lean_median_deg)) {
      out.push({ code: 'trunk', text: '가장 깊은 지점에서 몸통이 수직에서 ' + m.trunk.lean_median_deg + '도 기울었습니다.' });
    }
    if (Number.isFinite(m.consistency.depth_sd_deg) && m.consistency.depth_sd_deg >= 8) {
      out.push({ code: 'consistency', text: '반복마다 깊이 편차가 ' + m.consistency.depth_sd_deg + '도로 크게 나타났습니다.' });
    }
    return out;
  }

  function fingerprint(metrics) {
    const seed = [
      metrics.rep_count,
      metrics.depth.median_deg,
      metrics.depth.sd_deg,
      metrics.asymmetry.mean_deg,
      metrics.tempo.rep_median_s,
      metrics.trunk.lean_median_deg,
      metrics.frames,
      metrics.view,
    ].join('|');
    let hash = 0x811c9dc5;
    for (let i = 0; i < seed.length; i++) {
      hash ^= seed.charCodeAt(i);
      hash = (hash + ((hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24))) >>> 0;
    }
    return ('00000000' + hash.toString(16)).slice(-8);
  }

  const api = {
    VERSION,
    LANDMARK_INDEX,
    MIN_VISIBILITY,
    includedAngle,
    buildFrame,
    computeMetrics,
    describeMetrics,
    detectRepRuns,
    fingerprint,
    cameraViewLabel,
    percentile,
    median,
    sd,
  };

  if (typeof module === 'object' && module.exports) module.exports = api;
  global.SquatMetrics = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
