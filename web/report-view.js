/* Report rendering for the squat report (browser + Node).
 *
 * Everything the model writes goes through esc(). The fixed copy — measurement
 * limits, disclaimer, exercise how-to — comes from this file or the catalog,
 * never from the model, so a report can be re-rendered safely after the fact.
 */
(function (global) {
  'use strict';

  const CATALOG =
    (typeof require === 'function' && typeof module === 'object' && module.exports
      ? require('./exercise-catalog.js')
      : null) || global.SquatExercises || { get: () => null };

  const DISCLAIMER =
    '이 리포트는 한 대의 카메라로 추정한 관절 각도 수치를 운동 코칭 언어로 풀어 쓴 참고 자료입니다. 의료 진단·검사·치료가 아니며 정상·비정상을 판정하지 않습니다.';

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function num(value, suffix, digits) {
    if (value === null || value === undefined || value === '' || !Number.isFinite(Number(value))) return '—';
    const fixed = digits === undefined ? Number(value) : Number(value).toFixed(digits);
    return String(fixed) + (suffix || '');
  }

  const DEPTH_LABEL = { shallow: '얕은 편', moderate: '중간 깊이', deep: '깊은 편', very_deep: '매우 깊은 편', unknown: '판단 보류' };
  const SYM_LABEL = { small: '작음', moderate: '중간', large: '큼', unknown: '판단 보류' };
  const PRIORITY_LABEL = { high: '먼저', mid: '다음', low: '유지' };

  function chips(metrics) {
    return [
      { label: '스쿼트', value: num(metrics.rep_count, '회') },
      { label: '깊이', value: num(metrics.depth.median_deg, '°') },
      { label: '좌우 차이', value: num(metrics.asymmetry.mean_deg, '°') },
      { label: '템포', value: num(metrics.tempo.rep_median_s, '초', 1) },
    ];
  }

  function metricsTable(metrics) {
    const rows = [
      ['영상 길이', num(metrics.duration_s, '초', 1)],
      ['분석 프레임', num(metrics.frames_used) + ' / ' + num(metrics.frames)],
      ['촬영 각도', esc(metrics.view_label || '미지정')],
      ['각도 기준', metrics.depth.source === '3d' ? '3D 추정(골반 중심)' : '2D 투영(화면)'],
      ['가장 깊은 무릎 각도', num(metrics.depth.median_deg, '°')],
      ['2D 투영 무릎 각도', num(metrics.depth_2d_median_deg, '°')],
      ['깊이 편차', num(metrics.depth.sd_deg, '°')],
      ['좌우 차이(평균/최대)', num(metrics.asymmetry.mean_deg, '°') + ' / ' + num(metrics.asymmetry.max_deg, '°')],
      ['한 회 시간', num(metrics.tempo.rep_median_s, '초', 1)],
      ['내려가기 / 올라오기', num(metrics.tempo.descent_median_s, '초') + ' / ' + num(metrics.tempo.ascent_median_s, '초')],
      ['몸통 기울기', num(metrics.trunk.lean_median_deg, '°')],
      ['발 간격(어깨 대비)', num(metrics.stance.ratio_median, '배')],
    ];
    return (
      '<table class="sq-table"><tbody>' +
      rows
        .map((r) => '<tr><th>' + esc(r[0]) + '</th><td>' + r[1] + '</td></tr>')
        .join('') +
      '</tbody></table>'
    );
  }

  /* Free result screen summary + the paid teaser. The teaser shows only the
   * shape of what is inside, never the model's text. */
  function teaserHtml(metrics, options) {
    const opts = options || {};
    const price = opts.priceLabel || '4,900원';
    const locked = [
      '이번 수치가 운동 관점에서 무엇을 뜻하는지 3단락 해석',
      '깊이·좌우 차이·템포를 묶은 4주 진행 루틴 (주 3회)',
      '운동별 세트·횟수·큐 + 다음 측정 체크포인트',
    ];
    return (
      '<section class="sq-teaser">' +
      '<div class="sq-teaser-head"><span class="sq-badge">AI 정밀 리포트</span>' +
      '<h3>' +
      esc(metrics.rep_count) + '회 스쿼트, 수치는 여기까지 나왔습니다</h3></div>' +
      '<div class="sq-chips">' +
      chips(metrics)
        .map((c) => '<div class="sq-chip"><span>' + esc(c.label) + '</span><strong>' + esc(c.value) + '</strong></div>')
        .join('') +
      '</div>' +
      '<ul class="sq-locked">' +
      locked
        .map(
          (line) =>
            '<li><span class="sq-lock" aria-hidden="true">🔒</span><span>' + esc(line) + '</span></li>'
        )
        .join('') +
      '</ul>' +
      (metrics.flags && metrics.flags.length
        ? '<p class="sq-teaser-line">' + esc(metrics.flags[0].text) + '</p>'
        : '') +
      '<button class="sq-primary" type="button" data-sq-open-panel>' +
      '리포트 열기 · ' + esc(price) + '</button>' +
      '<p class="sq-micro">측정값은 이 기기 안에서만 계산됩니다. 영상은 어디에도 전송되지 않습니다.</p>' +
      '</section>'
    );
  }

  function findingsHtml(findings) {
    if (!Array.isArray(findings) || !findings.length) return '';
    return (
      '<ul class="sq-findings">' +
      findings
        .map(
          (f) =>
            '<li class="sq-finding sq-pri-' + esc(f.priority || 'mid') + '">' +
            '<div class="sq-finding-top"><span class="sq-pri">' + esc(PRIORITY_LABEL[f.priority] || '다음') + '</span>' +
            '<strong>' + esc(f.observation) + '</strong></div>' +
            '<p>' + esc(f.meaning) + '</p></li>'
        )
        .join('') +
      '</ul>'
    );
  }

  function exerciseBlock(step) {
    const ex = CATALOG.get ? CATALOG.get(step.exercise_id) : null;
    const name = esc(step.name || (ex && ex.name) || step.exercise_id);
    const how = ex ? esc(ex.how) : '';
    return (
      '<li class="sq-step">' +
      '<div class="sq-step-head"><strong>' + name + '</strong>' +
      '<span>' + esc(step.sets || '') + (step.sets && step.dose ? ' · ' : '') + esc(step.dose || '') + '</span></div>' +
      (step.cue ? '<p class="sq-cue">' + esc(step.cue) + '</p>' : '') +
      (how ? '<details><summary>하는 방법</summary><p>' + how + '</p></details>' : '') +
      '</li>'
    );
  }

  function weeksHtml(weeks) {
    if (!Array.isArray(weeks) || !weeks.length) return '';
    return (
      '<div class="sq-weeks">' +
      weeks
        .map(
          (w) =>
            '<article class="sq-week">' +
            '<header><span class="sq-week-no">' + esc(w.week) + '주차</span>' +
            '<strong>' + esc(w.focus || '') + '</strong></header>' +
            (w.weekly_goal ? '<p class="sq-goal">' + esc(w.weekly_goal) + '</p>' : '') +
            '<div class="sq-sessions">' +
            (w.sessions || [])
              .map(
                (s) =>
                  '<section class="sq-session"><h4>' + esc(s.day) + '일차 · ' + esc(s.minutes) + '분</h4>' +
                  '<ul>' + (s.steps || []).map(exerciseBlock).join('') + '</ul></section>'
              )
              .join('') +
            '</div></article>'
        )
        .join('') +
      '</div>'
    );
  }

  function listHtml(items, className) {
    if (!Array.isArray(items) || !items.length) return '';
    return '<ul class="' + className + '">' + items.map((i) => '<li>' + esc(typeof i === 'string' ? i : i.text || '') + '</li>').join('') + '</ul>';
  }

  function reportHtml(report, meta, metrics) {
    const m = meta || {};
    const r = report || {};
    const when = m.generatedAt ? new Date(m.generatedAt) : new Date();
    const stamp =
      when.getFullYear() + '.' + String(when.getMonth() + 1).padStart(2, '0') + '.' + String(when.getDate()).padStart(2, '0');
    return (
      '<article class="sq-report" id="sqReport">' +
      '<header class="sq-report-head">' +
      '<p class="sq-kicker">스쿼트 리포트 · 4주 루틴</p>' +
      '<h1>' + esc(r.headline || '스쿼트 리포트') + '</h1>' +
      '<p class="sq-lede">' + esc(r.summary_line || '') + '</p>' +
      '<p class="sq-meta">' + esc(stamp) + ' · 촬영 ' + esc((metrics && metrics.view_label) || m.view_label || '미지정') +
      ' · 스쿼트 ' + esc(metrics ? metrics.rep_count : '') + '회</p>' +
      '</header>' +
      (metrics ? '<section class="sq-block"><h2>이번 측정 수치</h2>' + metricsTable(metrics) + '</section>' : '') +
      '<section class="sq-block"><h2>해석</h2>' +
      (r.sections || [])
        .map((s) => '<div class="sq-section"><h3>' + esc(s.title) + '</h3><p>' + esc(s.body) + '</p></div>')
        .join('') +
      '</section>' +
      (r.findings && r.findings.length
        ? '<section class="sq-block"><h2>먼저 손볼 것</h2>' + findingsHtml(r.findings) + '</section>'
        : '') +
      '<section class="sq-block sq-block-routine"><h2>4주 루틴 <span class="sq-sub">주 3회 · 한 번 15분 안쪽</span></h2>' +
      weeksHtml(r.weeks) +
      '</section>' +
      (r.checkpoints && r.checkpoints.length
        ? '<section class="sq-block"><h2>체크포인트</h2>' + listHtml(r.checkpoints, 'sq-checklist') + '</section>'
        : '') +
      (r.cautions && r.cautions.length
        ? '<section class="sq-block sq-block-caution"><h2>주의</h2>' + listHtml(r.cautions, 'sq-cautions') + '</section>'
        : '') +
      '<section class="sq-block sq-block-limits"><h2>측정 한계</h2>' +
      listHtml((metrics && metrics.measurement_notes) || [], 'sq-notes') +
      '<p class="sq-disclaimer">' + esc(r.disclaimer || DISCLAIMER) + '</p>' +
      '</section>' +
      '<footer class="sq-report-foot">기하 계산: Kinelo × SUMediPose 오픈소스(MIT)의 관절각 정의를 사용했습니다. ' +
      '영상과 랜드마크는 이 기기에서만 처리되었습니다.</footer>' +
      '</article>'
    );
  }

  /* Sample data for the static preview build and for ?reportPreview=1. Values
   * are plausible for a 3-rep bodyweight squat filmed at 45 degrees; they are
   * clearly labelled as a sample in the UI, never passed off as a measurement. */
  function previewFixtures() {
    const metrics = {
      version: 1,
      view: 'angle45',
      view_label: '45도 사선',
      angle_source: 'world_3d',
      frames: 246,
      frames_used: 231,
      valid_ratio: 0.939,
      duration_s: 11.8,
      rep_count: 3,
      range_deg: 74.3,
      depth: { level: 'moderate', median_deg: 104.2, min_deg: 98.6, sd_deg: 5.1, source: '3d' },
      depth_2d_median_deg: 119.4,
      asymmetry: { level: 'moderate', mean_deg: 6.4, max_deg: 8.1 },
      tempo: { rep_median_s: 3.4, descent_median_s: 1.9, ascent_median_s: 1.2, rhythm: 1.58, sd_s: 0.3 },
      trunk: { lean_median_deg: 22.3, lean_max_deg: 27.5 },
      stance: { ratio_median: 1.28 },
      consistency: { depth_sd_deg: 5.1, tempo_sd_s: 0.3 },
      per_rep: [
        { index: 1, startMs: 1400, bottomMs: 3300, endMs: 4800, minKnee: 104.8, asymmetry: 5.2, descentS: 1.9, ascentS: 1.5, durationS: 3.4, trunk2d: 21.4, stance: 1.27 },
        { index: 2, startMs: 5400, bottomMs: 7100, endMs: 8700, minKnee: 98.6, asymmetry: 8.1, descentS: 1.7, ascentS: 1.6, durationS: 3.3, trunk2d: 27.5, stance: 1.3 },
        { index: 3, startMs: 9300, bottomMs: 11100, endMs: 12600, minKnee: 109.2, asymmetry: 5.9, descentS: 1.8, ascentS: 1.5, durationS: 3.3, trunk2d: 22.3, stance: 1.28 },
      ],
      quality: { ok: true, reasons: [] },
      measurement_notes: [
        '각도는 한 대의 카메라 영상에서 계산한 값입니다. 2D 각도는 화면 평면에 투영된 각도이고, 3D 각도는 포즈 모델이 추정한 골반 중심 좌표에서 계산한 값입니다.',
        '광학 모션캡처(Vicon 등)로 검증된 값이 아니며, 카메라 각도·거리·옷·가림에 따라 달라집니다.',
        '같은 자리·같은 각도에서 다시 재면 그 변화 비교에는 쓸 수 있습니다.',
      ],
      flags: [
        { code: 'depth', text: '가장 깊은 지점의 무릎 각도는 중간값 104.2도(중간 깊이)입니다.' },
        { code: 'asymmetry', text: '좌우 무릎 각도 차이는 평균 6.4도(중간)입니다.' },
      ],
      fingerprint: 'a1b2c3d4',
    };
    const report = {
      headline: '깊이는 충분합니다. 남은 건 좌우 균형과 회복 타이밍',
      summary_line: '3회 모두 무릎 각도가 100도 근처까지 내려왔고, 좌우 차이는 2회에서 8도로 조금 벌어졌습니다.',
      sections: [
        {
          title: '지금 수치에서 보이는 것',
          body:
            '세 번의 스쿼트가 모두 무릎 각도 99~109도 사이로 내려왔습니다. 편차가 5도 안쪽이라 깊이는 매번 비슷하게 나왔습니다. 한 번에 3.4초, 내려가는 구간이 1.9초, 올라오는 구간이 1.2초로 내려가는 쪽이 조금 더 깁니다. 몸통은 가장 깊은 지점에서 22도쯤 기울었습니다.',
        },
        {
          title: '운동 관점에서의 의미',
          body:
            '깊이 편차가 작다는 것은 같은 패턴을 반복해서 만들고 있다는 뜻입니다. 좌우 차이 6도는 두 번째 반복에서 8도까지 벌어졌는데, 이 정도는 한쪽 다리에 체중이 조금 더 실리거나 골반이 살짝 돌아가는 흐름에서 흔히 나타납니다. 몸통 기울기 22도는 무게 중심을 뒤로 두려는 자연스러운 보상입니다.',
        },
        {
          title: '4주 동안 어디부터',
          body:
            '1~2주는 발목과 고관절 가동성으로 바닥을 만들고, 같은 템포로 3초 내려가기를 넣습니다. 3주는 좌우를 따로 쓰는 스플릿 스쿼트로 차이를 줄이고, 4주에 다시 같은 자리에서 측정해 6도가 4도 아래로 내려오는지 봅니다. 세트 사이 휴식은 60초로 잡습니다.',
        },
      ],
      findings: [
        { metric: 'asymmetry', priority: 'high', observation: '2회 반복에서 좌우 차이 8.1도', meaning: '한쪽에 체중이 더 실리는 흐름입니다. 스플릿 스쿼트로 좌우를 따로 쓰는 연습을 먼저 넣습니다.' },
        { metric: 'tempo', priority: 'mid', observation: '내려가기 1.9초 / 올라오기 1.2초', meaning: '내려가는 속도가 조금 더 깁니다. 3초 내려가기로 리듬을 고정하면 깊이 편차가 더 줄어듭니다.' },
        { metric: 'trunk', priority: 'low', observation: '몸통 기울기 22도', meaning: '무게 중심을 뒤로 두는 자연스러운 각도입니다. 통증이 없다면 그대로 두고 발목 가동성만 보완합니다.' },
      ],
      weeks: [
        {
          week: 1,
          focus: '바닥 만들기',
          weekly_goal: '발목·고관절 가동성을 확보하고 같은 템포를 몸에 익힙니다.',
          sessions: [
            { day: 1, minutes: 14, steps: [{ exercise_id: 'ankle_rock', name: '발목 앞쪽 락', sets: '2세트', dose: '좌우 10회', cue: '뒤꿈치는 바닥에' }, { exercise_id: 'bodyweight_squat', name: '맨몸 스쿼트', sets: '3세트', dose: '10회', cue: '3초에 걸쳐 내려가기' }] },
            { day: 2, minutes: 13, steps: [{ exercise_id: 'hip_hinge', name: '힙 힌지 연습', sets: '2세트', dose: '10회', cue: '가슴은 앞을 향해' }, { exercise_id: 'glute_bridge', name: '글루트 브리지', sets: '2세트', dose: '12회', cue: '엉덩이로 밀기' }] },
            { day: 3, minutes: 15, steps: [{ exercise_id: 'tempo_squat', name: '3초 내려가기 스쿼트', sets: '3세트', dose: '6회', cue: '속도를 세며' }, { exercise_id: 'dead_bug', name: '데드버그', sets: '2세트', dose: '좌우 8회', cue: '허리 고정' }] },
          ],
        },
        {
          week: 2,
          focus: '깊이 고정',
          weekly_goal: '반복마다 깊이 편차 4도 안쪽으로 들어오게 합니다.',
          sessions: [
            { day: 1, minutes: 15, steps: [{ exercise_id: 'box_squat', name: '박스 스쿼트', sets: '3세트', dose: '8회', cue: '같은 높이에 닿기' }, { exercise_id: 'side_plank', name: '사이드 플랭크', sets: '2세트', dose: '좌우 20초', cue: '골반 처지지 않게' }] },
            { day: 2, minutes: 14, steps: [{ exercise_id: 'goblet_squat', name: '고블릿 스쿼트', sets: '3세트', dose: '8회', cue: '가슴 세우기' }, { exercise_id: 'clamshell', name: '클램쉘', sets: '2세트', dose: '좌우 12회', cue: '골반 고정' }] },
            { day: 3, minutes: 12, steps: [{ exercise_id: 'wall_sit', name: '월 싯 버티기', sets: '3세트', dose: '20초', cue: '무릎 90도' }, { exercise_id: 'thoracic_rotation', name: '흉추 회전', sets: '2세트', dose: '좌우 8회', cue: '등에서 돌리기' }] },
          ],
        },
        {
          week: 3,
          focus: '좌우 맞추기',
          weekly_goal: '한쪽씩 쓰는 동작으로 좌우 차이를 4도 아래로 줄입니다.',
          sessions: [
            { day: 1, minutes: 15, steps: [{ exercise_id: 'split_squat', name: '스플릿 스쿼트', sets: '2세트', dose: '좌우 8회', cue: '앞 무릎과 발끝 방향' }, { exercise_id: 'adductor_rock', name: '애덕터 락백', sets: '2세트', dose: '좌우 8회', cue: '3초 머물기' }] },
            { day: 2, minutes: 14, steps: [{ exercise_id: 'heel_elevated_squat', name: '뒤꿈치 받치고 스쿼트', sets: '2세트', dose: '8회', cue: '발목 제한 보완' }, { exercise_id: 'calf_raise_slow', name: '천천히 카프레이즈', sets: '2세트', dose: '12회', cue: '3초에 내리기' }] },
            { day: 3, minutes: 15, steps: [{ exercise_id: 'tempo_squat', name: '3초 내려가기 스쿼트', sets: '3세트', dose: '8회', cue: '좌우 체중 확인' }, { exercise_id: 'breath_brace', name: '호흡·브레이싱', sets: '1세트', dose: '5회 호흡', cue: '옆구리로 채우기' }] },
          ],
        },
        {
          week: 4,
          focus: '다시 재기',
          weekly_goal: '같은 자리에서 다시 측정해 좌우 차이와 깊이 편차를 비교합니다.',
          sessions: [
            { day: 1, minutes: 12, steps: [{ exercise_id: 'world_greatest_stretch', name: '월드 그레이티스트 스트레치', sets: '1세트', dose: '좌우 5회', cue: '천천히' }, { exercise_id: 'bodyweight_squat', name: '맨몸 스쿼트', sets: '2세트', dose: '8회', cue: '가볍게' }] },
            { day: 2, minutes: 15, steps: [{ exercise_id: 'goblet_squat', name: '고블릿 스쿼트', sets: '3세트', dose: '8회', cue: '같은 템포' }, { exercise_id: 'side_plank', name: '사이드 플랭크', sets: '2세트', dose: '좌우 25초', cue: '호흡 유지' }] },
            { day: 3, minutes: 10, steps: [{ exercise_id: 'bodyweight_squat', name: '맨몸 스쿼트', sets: '3세트', dose: '5회', cue: '측정 전 준비' }] },
          ],
        },
      ],
      checkpoints: [
        '2주차 끝: 반복마다 깊이 편차가 4도 안쪽인지 확인합니다.',
        '3주차 끝: 스플릿 스쿼트에서 좌우가 비슷하게 내려가는지 봅니다.',
        '4주차: 같은 자리·같은 각도에서 다시 찍어 좌우 차이와 깊이를 비교합니다.',
      ],
      cautions: [
        '무릎이나 허리에 통증이 생기면 그 세트는 멈추고 깊이를 줄이세요.',
        '통증·저림이 계속되면 운동을 미루고 전문가와 상담하세요.',
      ],
      disclaimer: DISCLAIMER,
    };
    return { metrics, report };
  }

  const api = {
    DISCLAIMER,
    esc,
    chips,
    metricsTable,
    teaserHtml,
    reportHtml,
    weeksHtml,
    findingsHtml,
    previewFixtures,
    PRIORITY_LABEL,
  };

  if (typeof module === 'object' && module.exports) module.exports = api;
  global.SquatReportView = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
