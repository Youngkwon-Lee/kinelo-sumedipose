'use strict';
// /api/report: code gate, input sanitising, output gate, rate limits, and the
// promise that no key, code list or prompt ever travels back to the browser.

const test = require('node:test');
const assert = require('node:assert/strict');
const api = require('../../api/report.js');
const { metricsFixture } = require('./fixtures.cjs');
const view = require('../../web/report-view.js');

const ENV = {
  OPENAI_API_KEY: 'sk-test-not-a-real-key',
  SQ_REPORT_ACCESS_CODES: 'SQ-7K2M-2411, SQ-AAAA-BBBB',
};

function res() {
  return {
    statusCode: 0,
    headers: {},
    body: null,
    setHeader(name, value) {
      this.headers[name] = value;
    },
    end(payload) {
      this.body = payload ? JSON.parse(payload) : null;
    },
  };
}

function completion(report) {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      choices: [{ message: { content: typeof report === 'string' ? report : JSON.stringify(report) } }],
      usage: { prompt_tokens: 2410, completion_tokens: 1980 },
    }),
  };
}

async function call(options) {
  const opts = options || {};
  const handler = api.createHandler({
    env: Object.assign({}, ENV, opts.env || {}),
    fetchImpl: opts.fetchImpl || (async () => completion(opts.report || view.previewFixtures().report)),
    now: opts.now,
  });
  const response = res();
  await handler(
    {
      method: opts.method || 'POST',
      body: opts.body === undefined ? body() : opts.body,
      headers: opts.headers || {},
    },
    response
  );
  return response;
}

function body(overrides) {
  return Object.assign(
    { code: 'SQ-7K2M-2411', metrics: metricsFixture({ asymmetryDeg: 7 }), goal: 'strength' },
    overrides || {}
  );
}

test('only POST is accepted', async () => {
  const r = await call({ method: 'GET' });
  assert.equal(r.statusCode, 405);
});

test('a missing key returns not_configured rather than calling upstream', async () => {
  const r = await call({ env: { OPENAI_API_KEY: '' }, body: body() });
  assert.equal(r.statusCode, 503);
  assert.equal(r.body.code, 'not_configured');
});

test('a wrong or missing access code never reaches the model', async () => {
  let called = 0;
  const fetchImpl = async () => {
    called += 1;
    return completion(view.previewFixtures().report);
  };
  const wrong = await call({ body: body({ code: 'SQ-0000-0000' }), fetchImpl });
  assert.equal(wrong.statusCode, 403);
  const missing = await call({ body: body({ code: '' }), fetchImpl });
  assert.equal(missing.statusCode, 403);
  assert.equal(called, 0);
});

test('access codes are compared without punctuation or case', async () => {
  const r = await call({ body: body({ code: 'sq 7k2m 2411' }) });
  assert.equal(r.statusCode, 200);
});

test('metrics that cannot carry a report are rejected before any charge', async () => {
  const metrics = metricsFixture();
  metrics.rep_count = 1;
  const r = await call({ body: body({ metrics }) });
  assert.equal(r.statusCode, 400);
  assert.equal(r.body.code, 'insufficient_measurement');
});

test('a happy path returns a report with the fixed disclaimer and 4 weeks', async () => {
  const r = await call({});
  assert.equal(r.statusCode, 200);
  assert.equal(r.body.ok, true);
  assert.equal(r.body.report.disclaimer, api.DISCLAIMER);
  assert.equal(r.body.report.weeks.length, 4);
  assert.deepEqual(
    r.body.report.weeks.map((w) => w.week),
    [1, 2, 3, 4]
  );
  assert.ok(r.body.report.weeks[0].sessions.length >= 2);
  assert.equal(r.body.report.weeks[0].sessions[0].steps[0].name, '발목 앞쪽 락');
  assert.equal(r.body.meta.model, 'gpt-6.1-terra');
  assert.ok(r.body.meta.cost_krw > 0);
});

test('the response never carries the key, the code list or the prompt', async () => {
  const r = await call({});
  const raw = JSON.stringify(r.body);
  assert.equal(raw.includes(ENV.OPENAI_API_KEY), false);
  assert.equal(raw.includes('7K2M2411'), false);
  assert.equal(raw.includes('너는 스쿼트'), false);
});

test('output with a medical claim is rejected', async () => {
  const report = JSON.parse(JSON.stringify(view.previewFixtures().report));
  report.sections[1].body = '이런 형태는 무릎 연골 질환에서 자주 보입니다.';
  const r = await call({ report });
  assert.equal(r.statusCode, 502);
  assert.equal(r.body.code, 'unsafe_output');
});

test('self-negating copy about diagnosis is not treated as a claim', async () => {
  const report = JSON.parse(JSON.stringify(view.previewFixtures().report));
  report.cautions.push('이 리포트는 의료 진단이 아닙니다.');
  const r = await call({ report });
  assert.equal(r.statusCode, 200);
});

test('a report missing a week is rejected instead of half-delivered', async () => {
  const report = JSON.parse(JSON.stringify(view.previewFixtures().report));
  report.weeks = report.weeks.slice(0, 3);
  const r = await call({ report });
  assert.equal(r.statusCode, 502);
  assert.equal(r.body.code, 'invalid_output');
});

test('unknown exercise ids are dropped from the routine', async () => {
  const report = JSON.parse(JSON.stringify(view.previewFixtures().report));
  report.weeks[0].sessions[0].steps.push({ exercise_id: 'invented_move', sets: '3세트', dose: '10회', cue: 'x' });
  const r = await call({ report });
  assert.equal(r.statusCode, 200);
  const ids = r.body.report.weeks[0].sessions[0].steps.map((s) => s.exercise_id);
  assert.equal(ids.includes('invented_move'), false);
});

test('upstream 429 becomes rate_limited, upstream 401 becomes not_configured', async () => {
  const limited = await call({
    fetchImpl: async () => ({ ok: false, status: 429, json: async () => ({}) }),
  });
  assert.equal(limited.statusCode, 429);
  const unauthorized = await call({
    fetchImpl: async () => ({ ok: false, status: 401, json: async () => ({}) }),
  });
  assert.equal(unauthorized.statusCode, 503);
});

test('a network failure becomes upstream_error', async () => {
  const r = await call({
    fetchImpl: async () => {
      throw new Error('boom');
    },
  });
  assert.equal(r.statusCode, 502);
  assert.equal(r.body.code, 'upstream_error');
});

test('refusal and unparseable content are handled', async () => {
  const refused = await call({
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ choices: [{ message: { refusal: 'no' } }] }) }),
  });
  assert.equal(refused.statusCode, 502);
  assert.equal(refused.body.code, 'refused');
  const garbage = await call({ report: 'not json' });
  assert.equal(garbage.statusCode, 502);
  assert.equal(garbage.body.code, 'invalid_output');
});

test('an oversized body is refused', async () => {
  const r = await call({ body: { code: 'SQ-7K2M-2411', metrics: { x: 'a'.repeat(60 * 1024) } } });
  assert.equal(r.statusCode, 413);
});

test('string bodies are parsed, malformed json is a bad request', async () => {
  const ok = await call({ body: JSON.stringify(body()) });
  assert.equal(ok.statusCode, 200);
  const bad = await call({ body: '{nope' });
  assert.equal(bad.statusCode, 400);
});

test('per-IP hourly limit stops the seventh call', async () => {
  let clock = 1_700_000_000_000;
  const handler = api.createHandler({
    env: ENV,
    fetchImpl: async () => completion(view.previewFixtures().report),
    now: () => clock,
  });
  const statuses = [];
  for (let i = 0; i < 7; i++) {
    const response = res();
    await handler({ method: 'POST', body: body(), headers: { 'x-forwarded-for': '1.2.3.4' } }, response);
    statuses.push(response.statusCode);
    clock += 1000;
  }
  assert.deepEqual(statuses.slice(0, 6), [200, 200, 200, 200, 200, 200]);
  assert.equal(statuses[6], 429);
});

test('sanitizeMetrics clamps impossible numbers', () => {
  const clean = api.sanitizeMetrics({
    rep_count: 999,
    frames_used: 100000,
    range_deg: 500,
    depth: { median_deg: 9999, level: 'weird' },
    asymmetry: { mean_deg: -50 },
    tempo: { rep_median_s: 999 },
    view: 'drone',
    goal: 'x',
  });
  assert.equal(clean.rep_count, 50);
  assert.equal(clean.depth.median_deg, 180);
  assert.equal(clean.depth.level, 'moderate');
  assert.equal(clean.asymmetry.mean_deg, 0);
  assert.equal(clean.tempo.rep_median_s, 60);
  assert.equal(clean.view, 'angle45');
});

test('sanitizeMetrics refuses a report with nothing to say', () => {
  assert.equal(api.sanitizeMetrics(null), null);
  assert.equal(api.sanitizeMetrics({ rep_count: 0 }), null);
});

test('the prompt lists only catalog ids and forbids diagnosis', () => {
  const prompt = api.systemPrompt('mobility');
  for (const id of Object.keys(api.EXERCISE_IDS)) assert.ok(prompt.includes(id), id);
  assert.ok(prompt.includes('진단하지 않는다'));
  assert.ok(prompt.includes('가동성'));
  const schema = api.schemaFor();
  assert.equal(schema.properties.weeks.items.properties.week.enum.length, 4);
  assert.equal(schema.properties.findings.items.properties.metric.enum.length, 6);
});

test('cost estimate follows the published price table', () => {
  const cost = api.estimateCostUsd('gpt-6.1-terra', { prompt_tokens: 2400, completion_tokens: 2000 });
  assert.ok(Math.abs(cost - 0.0124) < 1e-9);
  const cheap = api.estimateCostUsd('gpt-6-luna', { prompt_tokens: 2400, completion_tokens: 2000 });
  assert.ok(cheap < cost / 5);
});

test('the sample report in the preview satisfies the same output gate', () => {
  const checked = api.validateReport(view.previewFixtures().report);
  assert.equal(checked.ok, true, JSON.stringify(checked));
});
