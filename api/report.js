/* Squat report — paid AI report endpoint (Vercel Node function).
 *
 * Contract
 *   POST /api/report
 *   { code: "SQ-XXXX-XXXX", metrics: <numeric summary>, goal?: "strength"|"mobility"|"consistency" }
 *   -> { ok: true, report: <structured Korean report>, meta: { model, tokens, cost_krw } }
 *
 * Boundaries this file keeps:
 *  - No video, image, landmark, name or contact detail ever reaches this endpoint.
 *    Only the numeric summary the page already computed in the browser.
 *  - Nothing is stored or logged per buyer. Rate counters are per-instance memory
 *    and hold no payload.
 *  - The OpenAI key and the access codes live only in environment variables.
 *  - The report is a training document. The model is told not to diagnose, not to
 *    name injuries or conditions, and the server rejects output that breaks that
 *    rule. The disclaimer is fixed server copy, not model copy.
 *  - Numbers are camera estimates (see the repository README on what is and is not
 *    validated); the report is never presented as a clinical measurement.
 */
'use strict';

const CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const DEFAULT_MODEL = 'gpt-6.1-terra';
const CURRENCY_KRW = 1450; // planning assumption for the cost estimate only
const MAX_BODY_CHARS = 48 * 1024;
const MAX_OUTPUT_TOKENS = 6000;
const PER_HOUR = 6;
const PER_DAY = 15;
const GLOBAL_PER_DAY = 300;
const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;

const MIN_REPS = 2;
const MIN_FRAMES = 45;

const GOALS = ['strength', 'mobility', 'consistency'];
const GOAL_LABEL = { strength: '근력', mobility: '가동성', consistency: '꾸준함' };
const FINDING_METRICS = ['depth', 'asymmetry', 'tempo', 'trunk', 'consistency', 'stance'];
const PRIORITIES = ['high', 'mid', 'low'];

const DISCLAIMER =
  '이 리포트는 한 대의 카메라로 추정한 관절 각도 수치를 운동 코칭 언어로 풀어 쓴 참고 자료입니다. 의료 진단·검사·치료가 아니며 정상·비정상을 판정하지 않습니다.';

/* Canonical exercise ids. tests/web/catalog.test.cjs fails if this list and
 * web/exercise-catalog.js drift apart. */
const EXERCISE_IDS = {
  ankle_rock: '발목 앞쪽 락',
  hip_hinge: '힙 힌지 연습',
  goblet_squat: '고블릿 스쿼트',
  bodyweight_squat: '맨몸 스쿼트',
  tempo_squat: '3초 내려가기 스쿼트',
  box_squat: '박스 스쿼트',
  wall_sit: '월 싯 버티기',
  split_squat: '스플릿 스쿼트',
  heel_elevated_squat: '뒤꿈치 받치고 스쿼트',
  glute_bridge: '글루트 브리지',
  clamshell: '클램쉘',
  side_plank: '사이드 플랭크',
  dead_bug: '데드버그',
  adductor_rock: '애덕터 락백',
  thoracic_rotation: '흉추 회전',
  calf_raise_slow: '천천히 카프레이즈',
  breath_brace: '호흡·브레이싱',
  world_greatest_stretch: '월드 그레이티스트 스트레치',
};

// Model list prices in USD per 1M tokens (developers.openai.com/api/docs/pricing, 2026-11).
const PRICES = {
  'gpt-6.1-terra': { input: 1.0, output: 5.0 },
  'gpt-6.1-sol': { input: 2.0, output: 10.0 },
  'gpt-6-luna': { input: 0.1, output: 0.5 },
  'gpt-5.6-luna': { input: 0.2, output: 1.2 },
  'gpt-5.4-mini': { input: 0.75, output: 4.5 },
  'gpt-5-mini': { input: 0.25, output: 2.0 },
};

function estimateCostUsd(model, usage) {
  const price = PRICES[model] || PRICES[DEFAULT_MODEL];
  const input = Number(usage && usage.prompt_tokens) || 0;
  const output = Number(usage && usage.completion_tokens) || 0;
  return (input * price.input + output * price.output) / 1e6;
}

function normalizeCode(value) {
  return String(value == null ? '' : value).toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function parseCodes(raw) {
  return String(raw || '')
    .split(/[,\s]+/)
    .map(normalizeCode)
    .filter(Boolean);
}

function safeEqual(a, b) {
  const left = String(a);
  const right = String(b);
  let diff = left.length === right.length ? 0 : 1;
  const len = Math.max(left.length, right.length);
  for (let i = 0; i < len; i++) diff |= (left.charCodeAt(i) || 0) ^ (right.charCodeAt(i) || 0);
  return diff === 0;
}

function finiteOr(value, fallback = 0, lo = -1e9, hi = 1e9) {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(hi, Math.max(lo, n));
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

/* Keep only the numbers the page already shows, clamped to physical ranges.
 * Unknown keys are dropped so a modified client cannot smuggle instructions
 * into the prompt. */
function sanitizeMetrics(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const depth = raw.depth || {};
  const asym = raw.asymmetry || {};
  const tempo = raw.tempo || {};
  const trunk = raw.trunk || {};
  const stance = raw.stance || {};
  const consistency = raw.consistency || {};
  const metrics = {
    view: ['front', 'side', 'angle45'].includes(raw.view) ? raw.view : 'angle45',
    angle_source: raw.angle_source === 'image_2d' ? '2d' : '3d',
    frames: Math.round(finiteOr(raw.frames, 0, 0, 100000)),
    frames_used: Math.round(finiteOr(raw.frames_used, 0, 0, 100000)),
    valid_ratio: round1(finiteOr(raw.valid_ratio, 0, 0, 1) * 1000) / 1000,
    duration_s: round1(finiteOr(raw.duration_s, 0, 0, 3600)),
    rep_count: Math.round(finiteOr(raw.rep_count, 0, 0, 50)),
    range_deg: round1(finiteOr(raw.range_deg, 0, 0, 180)),
    depth: {
      level: ['shallow', 'moderate', 'deep', 'very_deep'].includes(depth.level) ? depth.level : 'moderate',
      median_deg: round1(finiteOr(depth.median_deg, 0, 0, 180)),
      min_deg: round1(finiteOr(depth.min_deg, 0, 0, 180)),
      sd_deg: round1(finiteOr(depth.sd_deg, 0, 0, 90)),
    },
    depth_2d_median_deg: round1(finiteOr(raw.depth_2d_median_deg, 0, 0, 180)),
    asymmetry: {
      mean_deg: round1(finiteOr(asym.mean_deg, 0, 0, 90)),
      max_deg: round1(finiteOr(asym.max_deg, 0, 0, 90)),
    },
    tempo: {
      rep_median_s: round1(finiteOr(tempo.rep_median_s, 0, 0, 60)),
      descent_median_s: round1(finiteOr(tempo.descent_median_s, 0, 0, 60)),
      ascent_median_s: round1(finiteOr(tempo.ascent_median_s, 0, 0, 60)),
    },
    trunk: {
      lean_median_deg: round1(finiteOr(trunk.lean_median_deg, 0, 0, 90)),
      lean_max_deg: round1(finiteOr(trunk.lean_max_deg, 0, 0, 90)),
    },
    stance: { ratio_median: round1(finiteOr(stance.ratio_median, 0, 0, 5) * 100) / 100 },
    consistency: {
      depth_sd_deg: round1(finiteOr(consistency.depth_sd_deg, 0, 0, 90)),
      tempo_sd_s: round1(finiteOr(consistency.tempo_sd_s, 0, 0, 60) * 100) / 100,
    },
  };
  // A report needs something to talk about.
  if (metrics.rep_count < MIN_REPS) return null;
  if (metrics.frames_used < MIN_FRAMES) return null;
  if (!(metrics.depth.median_deg > 0) || !(metrics.range_deg > 0)) return null;
  return metrics;
}

function sanitizeGoal(raw) {
  return GOALS.includes(raw) ? raw : 'strength';
}

function schemaFor() {
  const ids = Object.keys(EXERCISE_IDS);
  const step = {
    type: 'object',
    properties: {
      exercise_id: { type: 'string', enum: ids },
      sets: { type: 'string' },
      dose: { type: 'string' },
      cue: { type: 'string' },
    },
    required: ['exercise_id', 'sets', 'dose', 'cue'],
    additionalProperties: false,
  };
  const session = {
    type: 'object',
    properties: {
      day: { type: 'integer', enum: [1, 2, 3] },
      minutes: { type: 'integer' },
      steps: { type: 'array', items: step },
    },
    required: ['day', 'minutes', 'steps'],
    additionalProperties: false,
  };
  return {
    type: 'object',
    properties: {
      summary_line: { type: 'string' },
      headline: { type: 'string' },
      sections: {
        type: 'array',
        items: {
          type: 'object',
          properties: { title: { type: 'string' }, body: { type: 'string' } },
          required: ['title', 'body'],
          additionalProperties: false,
        },
      },
      findings: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            metric: { type: 'string', enum: FINDING_METRICS },
            priority: { type: 'string', enum: PRIORITIES },
            observation: { type: 'string' },
            meaning: { type: 'string' },
          },
          required: ['metric', 'priority', 'observation', 'meaning'],
          additionalProperties: false,
        },
      },
      weeks: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            week: { type: 'integer', enum: [1, 2, 3, 4] },
            focus: { type: 'string' },
            weekly_goal: { type: 'string' },
            sessions: { type: 'array', items: session },
          },
          required: ['week', 'focus', 'weekly_goal', 'sessions'],
          additionalProperties: false,
        },
      },
      checkpoints: { type: 'array', items: { type: 'string' } },
      cautions: { type: 'array', items: { type: 'string' } },
      disclaimer: { type: 'string' },
    },
    required: ['summary_line', 'headline', 'sections', 'findings', 'weeks', 'checkpoints', 'cautions', 'disclaimer'],
    additionalProperties: false,
  };
}

function systemPrompt(goal) {
  const list = Object.entries(EXERCISE_IDS)
    .map(([id, name]) => `- ${id} (${name})`)
    .join('\n');
  return [
    '너는 스쿼트 동작을 가르치는 운동 코치다. 사용자가 자기 카메라로 찍은 스쿼트 영상을 분석한 수치를 읽고, 4주 동안 혼자 할 수 있는 루틴을 정리해 준다.',
    '',
    '지켜야 할 선:',
    '1. 진단하지 않는다. "진단", "질환", "파열", "염증", "처방", "치료법", "디스크" 같은 의료 판정 표현을 쓰지 않는다. 정상/비정상 판정도 하지 않는다.',
    '2. 통증·수술·부상 이력은 알 수 없으므로 추정하지 않는다. 대신 cautions에 "통증이 있으면 멈추고 전문가와 상담"을 넣는다.',
    '3. 수치는 카메라 추정값이다. 소수점까지 인용하지 말고 반올림해 인용하고, "정확한 측정"이라고 말하지 않는다.',
    '4. 운동은 아래 목록의 id만 쓴다. 목록에 없는 운동을 만들지 않는다.',
    '5. 한국어 존댓말, 짧은 문장. 이모지·마크다운 기호 금지. 겁을 주지 않고, 효과를 보장하지 않는다.',
    '6. 좌우 차이가 크면 한쪽씩 하는 운동(스플릿 스쿼트 등)을 먼저 배치하고, 깊이가 얕으면 발목·고관절 가동성 운동을 먼저 둔다.',
    '7. 사용자가 고른 목표에 맞춰 운동 비중을 조정한다: ' + GOAL_LABEL[goal] + '.',
    '',
    '운동 목록(이 id만 사용):',
    list,
    '',
    '분량: sections 3개(각 300자 이내), findings 2~4개, weeks 정확히 4개(1~4주차, 각 주 3세션, 세션마다 2~3단계, 세션 10~16분), checkpoints 2~3개, cautions 1~3개, summary_line 70자 이내, headline 32자 이내.',
    'disclaimer는 다음 문장을 그대로 쓴다: "' + DISCLAIMER + '"',
  ].join('\n');
}

function userPrompt(metrics, goal) {
  const depthText = { shallow: '얕음', moderate: '중간', deep: '깊음', very_deep: '매우 깊음' }[metrics.depth.level];
  return [
    '아래는 이번 스쿼트 영상에서 계산한 요약 수치다. 각도는 무릎의 포함각(180도 = 다리를 폄)이고, 촬영 각도는 ' +
      ({ front: '정면', side: '측면', angle45: '45도 사선' }[metrics.view]) +
      '이다. 3D 값은 포즈 모델이 추정한 골반 중심 좌표 기준이다.',
    JSON.stringify(
      {
        metrics,
        depth_text: depthText,
        goal: GOAL_LABEL[goal],
        allowed_exercise_ids: Object.keys(EXERCISE_IDS),
      },
      null,
      1
    ),
    '',
    '이 사람이 이번 주에 바로 시작할 4주 루틴을 만들어라. 1주차에 가장 큰 문제 하나만 다루고, 4주차에는 같은 자리에서 다시 측정해 비교할 수 있게 끝맺어라.',
  ].join('\n');
}

/* Final gate. Structured outputs already match the schema; this rejects reports
 * whose content would be unsafe or unusable for a paying reader. */
function validateReport(raw) {
  if (!raw || typeof raw !== 'object') return { ok: false, code: 'invalid_output' };
  const text = (v, max) => (typeof v === 'string' && v.trim() ? v.trim().slice(0, max) : '');
  const report = {
    summary_line: text(raw.summary_line, 110),
    headline: text(raw.headline, 50),
    sections: [],
    findings: [],
    weeks: [],
    checkpoints: [],
    cautions: [],
    disclaimer: text(raw.disclaimer, 220),
  };
  if (!report.summary_line || !report.headline) return { ok: false, code: 'invalid_output' };

  for (const s of Array.isArray(raw.sections) ? raw.sections.slice(0, 4) : []) {
    const title = text(s && s.title, 28);
    const body = text(s && s.body, 700);
    if (title && body) report.sections.push({ title, body });
  }
  if (report.sections.length < 2) return { ok: false, code: 'invalid_output' };

  for (const f of Array.isArray(raw.findings) ? raw.findings.slice(0, 5) : []) {
    const metric = FINDING_METRICS.includes(f && f.metric) ? f.metric : null;
    const priority = PRIORITIES.includes(f && f.priority) ? f.priority : 'mid';
    const observation = text(f && f.observation, 120);
    const meaning = text(f && f.meaning, 220);
    if (metric && observation) report.findings.push({ metric, priority, observation, meaning });
  }

  const allowed = new Set(Object.keys(EXERCISE_IDS));
  for (const w of Array.isArray(raw.weeks) ? raw.weeks : []) {
    const weekNo = Number(w && w.week);
    if (!Number.isInteger(weekNo) || weekNo < 1 || weekNo > 4) continue;
    if (report.weeks.some((existing) => existing.week === weekNo)) continue;
    const sessions = [];
    for (const s of Array.isArray(w && w.sessions) ? w.sessions.slice(0, 3) : []) {
      const dayNo = Number(s && s.day);
      if (!Number.isInteger(dayNo) || dayNo < 1 || dayNo > 3) continue;
      const steps = [];
      for (const step of Array.isArray(s && s.steps) ? s.steps.slice(0, 3) : []) {
        const id = step && typeof step.exercise_id === 'string' ? step.exercise_id : '';
        if (!allowed.has(id)) continue;
        steps.push({
          exercise_id: id,
          name: EXERCISE_IDS[id],
          sets: text(step && step.sets, 24),
          dose: text(step && step.dose, 40),
          cue: text(step && step.cue, 120),
        });
      }
      if (!steps.length) continue;
      sessions.push({ day: dayNo, minutes: Math.round(finiteOr(s && s.minutes, 12, 6, 25)), steps });
    }
    sessions.sort((a, b) => a.day - b.day);
    if (sessions.length < 2) continue;
    report.weeks.push({
      week: weekNo,
      focus: text(w && w.focus, 40),
      weekly_goal: text(w && w.weekly_goal, 140),
      sessions,
    });
  }
  report.weeks.sort((a, b) => a.week - b.week);
  if (report.weeks.length < 4) return { ok: false, code: 'invalid_output' };

  for (const c of Array.isArray(raw.checkpoints) ? raw.checkpoints.slice(0, 3) : []) {
    const item = text(c, 160);
    if (item) report.checkpoints.push(item);
  }
  for (const c of Array.isArray(raw.cautions) ? raw.cautions.slice(0, 3) : []) {
    const item = text(c, 200);
    if (item) report.cautions.push(item);
  }
  if (!report.checkpoints.length) return { ok: false, code: 'invalid_output' };

  // The same words appear in legitimate self-negating copy ("의료 진단이 아닙니다"),
  // so strip those phrases before scanning for an actual claim.
  const NEGATED = [
    /(?:의료\s*)?진단(?:이|은|을)?\s*(?:아니|아님|아닙니다|목적이 아님)/g,
    /진단\s*[·/]\s*검사/g,
    /진단하지 않/g,
    /치료(?:가|는)?\s*아니/g,
  ];
  function stripSelfNegation(text) {
    let out = String(text);
    for (const pattern of NEGATED) out = out.replace(pattern, ' ');
    return out;
  }
  const banned = /(진단|질환|파열|염증|처방|치료법|디스크|정상\s*\/\s*비정상 판정)/;
  const visible = [
    report.headline,
    report.summary_line,
    ...report.sections.map((s) => s.body),
    ...report.findings.map((f) => f.observation + f.meaning),
  ].join(' ');
  if (banned.test(stripSelfNegation(visible))) return { ok: false, code: 'unsafe_output' };

  // The disclaimer is fixed copy: never let the model rewrite it away.
  report.disclaimer = DISCLAIMER;
  if (!report.cautions.length) report.cautions.push('통증이나 저림이 있으면 그 세트를 멈추고 전문가와 상담하세요.');
  return { ok: true, report };
}

function sendJson(res, status, body) {
  res.statusCode = status;
  if (typeof res.setHeader === 'function') {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
  }
  res.end(JSON.stringify(body));
}

function createHandler(options = {}) {
  const env = options.env || process.env;
  const fetchImpl = options.fetchImpl || (typeof fetch === 'function' ? fetch : null);
  const now = options.now || Date.now;
  const hits = new Map(); // ip -> { hourly: [t], daily: [t] }
  let globalDay = { start: 0, count: 0 };

  function limitedByIp(ip) {
    const t = now();
    const entry = hits.get(ip) || { hourly: [], daily: [] };
    entry.hourly = entry.hourly.filter((at) => t - at < HOUR_MS);
    entry.daily = entry.daily.filter((at) => t - at < DAY_MS);
    const blocked = entry.hourly.length >= PER_HOUR || entry.daily.length >= PER_DAY;
    if (!blocked) {
      entry.hourly.push(t);
      entry.daily.push(t);
    }
    hits.set(ip, entry);
    if (hits.size > 5000) hits.clear();
    return blocked;
  }

  function limitedGlobally() {
    const t = now();
    if (t - globalDay.start > DAY_MS) globalDay = { start: t, count: 0 };
    if (globalDay.count >= GLOBAL_PER_DAY) return true;
    globalDay.count += 1;
    return false;
  }

  return async function handler(req, res) {
    if (req.method !== 'POST') return sendJson(res, 405, { ok: false, code: 'method_not_allowed' });
    const apiKey = env.OPENAI_API_KEY;
    if (!apiKey) return sendJson(res, 503, { ok: false, code: 'not_configured' });
    if (!fetchImpl) return sendJson(res, 503, { ok: false, code: 'not_configured' });

    let body = req.body;
    if (typeof body === 'string') {
      if (body.length > MAX_BODY_CHARS) return sendJson(res, 413, { ok: false, code: 'body_too_large' });
      try {
        body = JSON.parse(body);
      } catch (_) {
        return sendJson(res, 400, { ok: false, code: 'bad_request' });
      }
    }
    if (!body || typeof body !== 'object') return sendJson(res, 400, { ok: false, code: 'bad_request' });
    if (JSON.stringify(body).length > MAX_BODY_CHARS) return sendJson(res, 413, { ok: false, code: 'body_too_large' });

    const codes = parseCodes(env.SQ_REPORT_ACCESS_CODES);
    const given = normalizeCode(body.code);
    if (!codes.length || !given || !codes.some((code) => safeEqual(code, given))) {
      return sendJson(res, 403, { ok: false, code: 'invalid_code' });
    }

    const metrics = sanitizeMetrics(body.metrics);
    if (!metrics) return sendJson(res, 400, { ok: false, code: 'insufficient_measurement' });
    const goal = sanitizeGoal(body.goal);

    const ip = String(
      (req.headers && (req.headers['x-forwarded-for'] || req.headers['x-real-ip'])) || 'unknown'
    )
      .split(',')[0]
      .trim();
    if (limitedGlobally() || limitedByIp(ip)) return sendJson(res, 429, { ok: false, code: 'rate_limited' });

    const model = env.SQ_REPORT_MODEL || DEFAULT_MODEL;
    const payload = {
      model,
      messages: [
        { role: 'system', content: systemPrompt(goal) },
        { role: 'user', content: userPrompt(metrics, goal) },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: { name: 'squat_report', strict: true, schema: schemaFor() },
      },
      max_completion_tokens: MAX_OUTPUT_TOKENS,
    };

    let upstream;
    try {
      upstream = await fetchImpl(CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + apiKey },
        body: JSON.stringify(payload),
      });
    } catch (_) {
      return sendJson(res, 502, { ok: false, code: 'upstream_error' });
    }

    let data = null;
    try {
      data = await upstream.json();
    } catch (_) {
      data = null;
    }
    if (!upstream.ok || !data) {
      const status = upstream.status === 429 ? 429 : upstream.status === 401 ? 503 : 502;
      return sendJson(res, status, {
        ok: false,
        code: status === 429 ? 'rate_limited' : status === 503 ? 'not_configured' : 'upstream_error',
      });
    }

    const choice = data.choices && data.choices[0];
    const content = choice && choice.message && choice.message.content;
    if (choice && choice.message && choice.message.refusal) {
      return sendJson(res, 502, { ok: false, code: 'refused' });
    }
    let parsed;
    try {
      parsed = typeof content === 'string' ? JSON.parse(content) : content;
    } catch (_) {
      parsed = null;
    }
    const checked = validateReport(parsed);
    if (!checked.ok) return sendJson(res, 502, { ok: false, code: checked.code });

    const usage = data.usage || {};
    const costUsd = estimateCostUsd(model, usage);
    return sendJson(res, 200, {
      ok: true,
      report: checked.report,
      meta: {
        model,
        tokens: { input: usage.prompt_tokens || 0, output: usage.completion_tokens || 0 },
        cost_usd: Math.round(costUsd * 1e6) / 1e6,
        cost_krw: Math.round(costUsd * CURRENCY_KRW),
        generatedAt: new Date(now()).toISOString(),
      },
    });
  };
}

module.exports = createHandler();
module.exports.createHandler = createHandler;
module.exports.validateReport = validateReport;
module.exports.sanitizeMetrics = sanitizeMetrics;
module.exports.sanitizeGoal = sanitizeGoal;
module.exports.normalizeCode = normalizeCode;
module.exports.schemaFor = schemaFor;
module.exports.systemPrompt = systemPrompt;
module.exports.userPrompt = userPrompt;
module.exports.DISCLAIMER = DISCLAIMER;
module.exports.EXERCISE_IDS = EXERCISE_IDS;
module.exports.GOALS = GOALS;
module.exports.PRICES = PRICES;
module.exports.estimateCostUsd = estimateCostUsd;
