'use strict';
// The exercise catalog is the single source of ids used by the browser; the API
// keeps its own copy for the prompt and for validation. They must not drift.

const test = require('node:test');
const assert = require('node:assert/strict');
const catalog = require('../../web/exercise-catalog.js');
const api = require('../../api/report.js');

test('browser catalog and API id list are identical', () => {
  assert.deepEqual(catalog.ids().slice().sort(), Object.keys(api.EXERCISE_IDS).sort());
});

test('every id is a stable slug and every name matches', () => {
  for (const ex of catalog.EXERCISES) {
    assert.match(ex.id, /^[a-z][a-z0-9_]{2,30}$/);
    assert.equal(api.EXERCISE_IDS[ex.id], ex.name);
  }
});

test('each exercise carries the how-to the report renders', () => {
  for (const ex of catalog.EXERCISES) {
    assert.ok(ex.name && ex.name.length <= 24, ex.id);
    assert.ok(ex.how && ex.how.length > 10, ex.id);
    assert.ok(ex.cue && ex.cue.length > 3, ex.id);
    assert.ok(ex.category);
  }
});

test('ids are unique', () => {
  const ids = catalog.ids();
  assert.equal(new Set(ids).size, ids.length);
});
