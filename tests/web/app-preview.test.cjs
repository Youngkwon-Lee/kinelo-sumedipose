'use strict';
// Page controller smoke test without a browser: a tiny DOM stub, the real
// app.js, and the ?reportPreview=1 route, which renders the free result and a
// sample report through the same code the buyer-facing page uses.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..', '..');

function makeNode(id) {
  return {
    id,
    innerHTML: '',
    textContent: '',
    hidden: false,
    className: '',
    style: {},
    value: '',
    disabled: false,
    files: [],
    width: 0,
    height: 0,
    videoWidth: 720,
    videoHeight: 1280,
    duration: 0,
    ended: false,
    currentTime: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener() {},
    removeEventListener() {},
    removeAttribute() {},
    setAttribute() {},
    querySelector: () => null,
    querySelectorAll: () => [],
    appendChild(c) {
      return c;
    },
    pause() {},
    play() {
      return Promise.resolve();
    },
    getContext: () => ({
      clearRect() {},
      beginPath() {},
      moveTo() {},
      lineTo() {},
      stroke() {},
      fill() {},
      arc() {},
      set strokeStyle(v) {},
      set fillStyle(v) {},
      set lineWidth(v) {},
    }),
  };
}

function boot(search) {
  const nodes = new Map();
  for (const id of [
    'screen-start', 'screen-analyze', 'screen-result', 'screen-access', 'screen-report',
    'video', 'overlay', 'countdown', 'progress-bar', 'analyze-status', 'start-status',
    'result-meta', 'result-chips', 'result-flags', 'teaser-slot', 'quality-block',
    'btn-camera', 'file-input', 'btn-cancel', 'btn-retry', 'btn-close-access',
    'buy-link', 'buy-price', 'code-input', 'btn-submit', 'access-status', 'code-form',
    'btn-print', 'btn-new', 'report-slot',
  ]) {
    const node = makeNode(id);
    // Mirrors the markup: every screen except the start screen begins hidden.
    if (/^screen-/.test(id)) node.hidden = id !== 'screen-start';
    nodes.set(id, node);
  }
  const screens = ['screen-start', 'screen-analyze', 'screen-result', 'screen-access', 'screen-report'].map((id) => nodes.get(id));
  const document = {
    readyState: 'complete',
    getElementById: (id) => nodes.get(id) || null,
    querySelectorAll: (sel) => (sel === '.sq-screen' ? screens : []),
    querySelector: () => null,
    addEventListener() {},
    createElement: (tag) => makeNode(tag),
  };
  const storage = new Map();
  const ctx = {
    document,
    console,
    Date,
    JSON,
    Math,
    Number,
    String,
    Object,
    Array,
    Promise,
    Error,
    isFinite,
    setInterval,
    clearInterval,
    setTimeout,
    performance: { now: () => 0 },
    location: { search },
    navigator: {},
    URL: { createObjectURL: () => 'blob:x', revokeObjectURL() {} },
    isSecureContext: true,
    localStorage: {
      getItem: (k) => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, v),
      removeItem: (k) => storage.delete(k),
    },
    fetch: async () => ({ ok: false, status: 500, json: async () => ({}) }),
    print() {},
    scrollTo() {},
    addEventListener() {},
    removeEventListener() {},
  };
  ctx.window = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  for (const file of ['web/config.js', 'web/squat-metrics.js', 'web/exercise-catalog.js', 'web/report-view.js', 'web/app.js']) {
    vm.runInContext(fs.readFileSync(path.join(ROOT, file), 'utf8'), ctx, { filename: file });
  }
  return { nodes, screens, ctx };
}

test('the preview route renders the free result and a sample report', () => {
  const { nodes, screens } = boot('?reportPreview=1');
  const teaser = nodes.get('teaser-slot').innerHTML;
  const report = nodes.get('report-slot').innerHTML;
  assert.ok(teaser.includes('sq-teaser'), 'teaser missing');
  assert.ok(teaser.includes('4,900원'), 'price missing');
  assert.ok(report.includes('4주 루틴'), 'routine missing');
  assert.ok(report.includes('측정 한계'), 'limits missing');
  assert.ok(nodes.get('result-chips').innerHTML.includes('104.2'));
  assert.equal(nodes.get('result-meta').textContent.includes('45도 사선'), true);
  const visible = screens.filter((s) => !s.hidden).map((s) => s.id);
  assert.deepEqual(visible, ['screen-result']);
});

test('without the preview flag the start screen is the entry point', () => {
  const { screens, nodes } = boot('');
  const visible = screens.filter((s) => !s.hidden).map((s) => s.id);
  assert.deepEqual(visible, ['screen-start']);
  assert.equal(nodes.get('teaser-slot').innerHTML, '');
});
