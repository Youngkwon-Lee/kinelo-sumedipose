// stdin: [[p1, p2, p3], ...]  ->  stdout: { angles: [...], indices: {...} }
// Used by tests/test_web_metrics_parity.py to compare the browser implementation
// with src/sumedipose/geometry.py on the same inputs.
'use strict';

const M = require('../../web/squat-metrics.js');

let input = '';
process.stdin.on('data', (chunk) => {
  input += chunk;
});
process.stdin.on('end', () => {
  const triples = JSON.parse(input);
  const angles = triples.map((triple) => M.includedAngle(triple[0], triple[1], triple[2]));
  process.stdout.write(JSON.stringify({ angles, indices: M.LANDMARK_INDEX }));
});
