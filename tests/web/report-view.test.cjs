'use strict';
// Rendering: model text is escaped, the free teaser shows no paid content, and
// the paid report renders the metrics, the 4-week routine and the fixed limits.

const test = require('node:test');
const assert = require('node:assert/strict');
const view = require('../../web/report-view.js');
const M = require('../../web/squat-metrics.js');
const { metricsFixture } = require('./fixtures.cjs');

test('esc neutralises markup from the model', () => {
  assert.equal(view.esc('<img src=x onerror=alert(1)>'), '&lt;img src=x onerror=alert(1)&gt;');
  assert.equal(view.esc('"quote" & \'apos\''), '&quot;quote&quot; &amp; &#39;apos&#39;');
});

test('the teaser shows the free numbers and no report body', () => {
  const metrics = metricsFixture({ asymmetryDeg: 7 });
  const html = view.teaserHtml(metrics, { priceLabel: '4,900원' });
  assert.ok(html.includes('4,900원'));
  assert.ok(html.includes('스쿼트'));
  assert.ok(html.includes('sq-locked'));
  assert.equal(html.includes('4주 루틴'), false);
  assert.equal(html.includes('<script'), false);
});

test('the paid report renders metrics, weeks, exercise how-to and the disclaimer', () => {
  const { metrics, report } = view.previewFixtures();
  const html = view.reportHtml(report, { generatedAt: '2026-11-18T12:00:00Z' }, metrics);
  assert.ok(html.includes('4주 루틴'));
  for (const week of [1, 2, 3, 4]) assert.ok(html.includes(week + '주차'), String(week));
  assert.ok(html.includes('발목 앞쪽 락'));
  assert.ok(html.includes('하는 방법'));
  assert.ok(html.includes(view.DISCLAIMER));
  assert.ok(html.includes('104.2'));
  assert.ok(html.includes('2026.11.18'));
});

test('model text cannot inject markup into the report', () => {
  const { metrics, report } = view.previewFixtures();
  const hostile = JSON.parse(JSON.stringify(report));
  hostile.headline = '<script>alert(1)</script>';
  hostile.sections[0].body = '<iframe src="//evil"></iframe>';
  const html = view.reportHtml(hostile, {}, metrics);
  assert.equal(html.includes('<script>alert(1)</script>'), false);
  assert.equal(html.includes('<iframe'), false);
  assert.ok(html.includes('&lt;script&gt;'));
});

test('an unknown exercise id degrades to the raw id, not a crash', () => {
  const html = view.weeksHtml([
    { week: 1, focus: 'x', weekly_goal: 'y', sessions: [{ day: 1, minutes: 12, steps: [{ exercise_id: 'mystery', sets: '2', dose: '5', cue: 'z' }] }] },
  ]);
  assert.ok(html.includes('mystery'));
});

test('preview fixtures stay consistent with the metrics engine', () => {
  const { metrics } = view.previewFixtures();
  assert.equal(metrics.quality.ok, true);
  assert.equal(metrics.depth.source, '3d');
  assert.equal(M.fingerprint(metrics).length, 8);
});
