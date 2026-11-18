/* Exercise catalog for the squat report.
 *
 * The paid report may only prescribe these ids: the API validates every step
 * against this list (checked by tests/web/catalog.test.cjs), and the report
 * view expands each id into the short how-to below. Adding an exercise means
 * adding it here AND in the EXERCISE_IDS list of api/report.js.
 */
(function (global) {
  'use strict';

  const EXERCISES = [
    {
      id: 'ankle_rock',
      name: '발목 앞쪽 락',
      category: '가동성',
      target: '발목',
      how: '벽을 짚고 한 발을 반 발짝 뒤로 둡니다. 뒤꿈치를 바닥에 붙인 채 무릎을 벽 쪽으로 천천히 밀었다 놓습니다.',
      cue: '발 안쪽 아치가 무너지지 않게, 뒤꿈치는 계속 바닥에.',
      default_dose: '좌우 10회',
    },
    {
      id: 'hip_hinge',
      name: '힙 힌지 연습',
      category: '패턴',
      target: '고관절',
      how: '막대기나 손잡이를 등에 세로로 대고 섭니다. 무릎은 살짝 굽힌 채 엉덩이를 뒤로 보내 상체를 숙였다가 돌아옵니다.',
      cue: '허리가 말리지 않게, 가슴은 앞을 향해.',
      default_dose: '2세트 10회',
    },
    {
      id: 'goblet_squat',
      name: '고블릿 스쿼트',
      category: '주운동',
      target: '하체 전체',
      how: '가벼운 덤벨이나 물병을 가슴 앞에 안고 섭니다. 그 무게를 안은 채 앉았다 일어납니다.',
      cue: '가슴을 세우고, 무릎은 발가락 방향으로.',
      default_dose: '3세트 8회',
    },
    {
      id: 'bodyweight_squat',
      name: '맨몸 스쿼트',
      category: '주운동',
      target: '하체 전체',
      how: '두 발을 어깨너비로 두고 앉았다 일어납니다. 손은 가슴 앞에서 균형을 잡습니다.',
      cue: '발바닥 전체가 바닥에 붙어 있는지 확인합니다.',
      default_dose: '3세트 10회',
    },
    {
      id: 'tempo_squat',
      name: '3초 내려가기 스쿼트',
      category: '주운동',
      target: '하체 전체·조절',
      how: '맨몸 스쿼트를 3초에 걸쳐 내려가고 1초에 올라옵니다. 내려가는 동안 멈추지 않고 일정한 속도를 유지합니다.',
      cue: '속도를 세며 내려가면 깊이가 고르게 나옵니다.',
      default_dose: '3세트 6회',
    },
    {
      id: 'box_squat',
      name: '박스 스쿼트',
      category: '주운동',
      target: '깊이 조준',
      how: '뒤에 의자나 낮은 박스를 두고, 그 위에 엉덩이를 살짝 닿혔다가 일어납니다.',
      cue: '앉듯이 내려가되 엉덩이를 떨어뜨리지 않습니다.',
      default_dose: '3세트 8회',
    },
    {
      id: 'wall_sit',
      name: '월 싯 버티기',
      category: '주운동',
      target: '무릎 주변 지구력',
      how: '등을 벽에 대고 미끄러져 내려가 무릎이 90도쯤 되면 그 자리에서 버팁니다.',
      cue: '무릎이 발가락보다 지나치게 앞으로 나가지 않게.',
      default_dose: '20초 3회',
    },
    {
      id: 'split_squat',
      name: '스플릿 스쿼트',
      category: '주운동',
      target: '한쪽 다리·균형',
      how: '한 발을 앞, 한 발을 뒤로 두고 두 무릎을 굽혀 내려갔다 옵니다.',
      cue: '앞 무릎과 발끝 방향을 맞춥니다.',
      default_dose: '좌우 8회 2세트',
    },
    {
      id: 'heel_elevated_squat',
      name: '뒤꿈치 받치고 스쿼트',
      category: '보조',
      target: '발목 제한 보완',
      how: '얇은 책이나 웨지를 발뒤꿈치 아래에 두고 맨몸 스쿼트를 합니다.',
      cue: '받침을 써도 무릎은 발끝 방향으로.',
      default_dose: '2세트 8회',
    },
    {
      id: 'glute_bridge',
      name: '글루트 브리지',
      category: '보조',
      target: '엉덩이',
      how: '누워서 무릎을 세우고 엉덩이를 들어 몸통과 허벅지가 일직선이 되게 합니다.',
      cue: '허리가 아니라 엉덩이로 밀어 올립니다.',
      default_dose: '2세트 12회',
    },
    {
      id: 'clamshell',
      name: '클램쉘',
      category: '보조',
      target: '엉덩이 옆',
      how: '옆으로 누워 무릎을 굽히고, 발을 붙인 채 위쪽 무릎을 천천히 벌렸다 모읍니다.',
      cue: '골반이 뒤로 돌아가지 않게.',
      default_dose: '좌우 12회',
    },
    {
      id: 'side_plank',
      name: '사이드 플랭크',
      category: '보조',
      target: '몸통 옆',
      how: '옆으로 누워 팔꿈치로 바닥을 누르며 허리를 들어 몸을 일직선으로 만듭니다.',
      cue: '숨을 참지 말고, 골반이 아래로 처지지 않게.',
      default_dose: '좌우 20초 2세트',
    },
    {
      id: 'dead_bug',
      name: '데드버그',
      category: '보조',
      target: '몸통 앞·조절',
      how: '누워서 팔과 무릎을 세운 뒤, 반대쪽 팔과 다리를 천천히 펴며 내렸다 돌아옵니다.',
      cue: '허리가 바닥에서 뜨지 않는 범위까지만.',
      default_dose: '좌우 8회',
    },
    {
      id: 'adductor_rock',
      name: '애덕터 락백',
      category: '가동성',
      target: '안쪽 허벅지',
      how: '네발 자세에서 한쪽 다리를 옆으로 뻗고, 엉덩이를 뒤로 보내 그 다리 안쪽을 늘립니다.',
      cue: '천천히 뒤로 갔다 3초 머뭅니다.',
      default_dose: '좌우 8회',
    },
    {
      id: 'thoracic_rotation',
      name: '흉추 회전',
      category: '가동성',
      target: '등·어깨',
      how: '옆으로 누워 위쪽 팔을 크게 돌려 등 뒤 바닥을 짚습니다.',
      cue: '허리는 고정하고 등에서 돌립니다.',
      default_dose: '좌우 8회',
    },
    {
      id: 'calf_raise_slow',
      name: '천천히 카프레이즈',
      category: '보조',
      target: '종아리·발목',
      how: '발 앞쪽으로 서서 뒤꿈치를 천천히 올렸다 3초에 걸쳐 내립니다.',
      cue: '벽을 짚어 균형을 잡아도 됩니다.',
      default_dose: '2세트 12회',
    },
    {
      id: 'breath_brace',
      name: '호흡·브레이싱',
      category: '준비',
      target: '몸통 압력',
      how: '서서 숨을 들이마시며 배와 옆구리를 팽팽하게 만든 뒤, 그 상태로 숨을 내쉬며 버팁니다.',
      cue: '배를 안으로 빨아들이지 말고 옆으로 채웁니다.',
      default_dose: '5회 호흡',
    },
    {
      id: 'world_greatest_stretch',
      name: '월드 그레이티스트 스트레치',
      category: '준비',
      target: '고관절·등',
      how: '한 발을 앞에 두고 깊게 앉아 팔꿈치를 바닥 쪽으로 내린 뒤, 위쪽 팔을 하늘로 돌립니다.',
      cue: '몸이 흔들리면 발을 조금 넓혀 잡습니다.',
      default_dose: '좌우 5회',
    },
  ];

  const BY_ID = {};
  for (const ex of EXERCISES) BY_ID[ex.id] = ex;

  const api = {
    EXERCISES,
    BY_ID,
    ids: () => EXERCISES.map((e) => e.id),
    get: (id) => BY_ID[id] || null,
    categories: () => Array.from(new Set(EXERCISES.map((e) => e.category))),
  };

  if (typeof module === 'object' && module.exports) module.exports = api;
  global.SquatExercises = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
