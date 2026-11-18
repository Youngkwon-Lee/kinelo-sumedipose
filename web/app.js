/* Squat report — page controller.
 *
 * Flow: camera or video file -> MediaPipe pose landmarks per frame -> the
 * browser computes the metric summary (web/squat-metrics.js) -> free result ->
 * (optional, code-gated) POST /api/report -> paid 4-week report.
 *
 * Privacy: the video never leaves the device and is not stored. The only
 * network call this page makes for the report carries the numeric summary.
 */
(function () {
  'use strict';

  const cfg = window.SQUAT_REPORT_CONFIG || {};
  const M = window.SquatMetrics;
  const V = window.SquatReportView;

  const $ = (id) => document.getElementById(id);
  const state = {
    frames: [],
    metrics: null,
    view: 'angle45',
    goal: 'strength',
    landmarker: null,
    stream: null,
    objectUrl: '',
    timer: null,
    cancelled: false,
    finished: false,
    report: null,
    meta: null,
  };

  const QUALITY_TEXT = {
    too_few_frames: '영상이 너무 짧습니다. 5초 이상, 스쿼트 2회 이상 담아 주세요.',
    low_visibility: '몸 전체가 화면에 들어오지 않았습니다. 3~4걸음 떨어져 전신이 보이게 찍어 주세요.',
    no_rep_detected: '스쿼트 동작을 찾지 못했습니다. 평소보다 조금 더 깊게 3회 해 주세요.',
    small_range: '무릎이 충분히 굽혀지지 않았습니다. 조금 더 앉았다 일어나 주세요.',
    too_short: '측정 시간이 짧습니다. 5초 이상 찍어 주세요.',
  };

  const ERROR_TEXT = {
    invalid_code: '이용 코드를 확인해 주세요.',
    insufficient_measurement: '측정값이 리포트를 만들 만큼 충분하지 않습니다. 다시 찍어 주세요.',
    rate_limited: '요청이 너무 잦습니다. 잠시 후 다시 시도해 주세요.',
    not_configured: '리포트 기능이 아직 준비되지 않았습니다.',
    upstream_error: '리포트 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.',
    unsafe_output: '리포트를 만들지 못했습니다. 다시 시도해 주세요.',
    invalid_output: '리포트를 만들지 못했습니다. 다시 시도해 주세요.',
    refused: '리포트를 만들지 못했습니다. 다시 시도해 주세요.',
    bad_request: '요청을 처리하지 못했습니다.',
  };

  function show(screen) {
    for (const el of document.querySelectorAll('.sq-screen')) el.hidden = el.id !== screen;
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  function status(el, text, kind) {
    if (!el) return;
    el.textContent = text || '';
    el.className = 'sq-status' + (kind ? ' is-' + kind : '');
  }

  /* ---------------------------------------------------------------- pose --- */

  async function ensureLandmarker() {
    if (state.landmarker) return state.landmarker;
    const vision = await import(cfg.visionUrl + '/vision_bundle.mjs');
    const fileset = await vision.FilesetResolver.forVisionTasks(cfg.wasmUrl);
    const modelAssetPath = (cfg.modelUrls && cfg.modelUrls[cfg.poseModel]) || cfg.modelUrls.full;
    const options = {
      baseOptions: { modelAssetPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
    };
    let landmarker;
    try {
      landmarker = await vision.PoseLandmarker.createFromOptions(fileset, options);
    } catch (err) {
      // Some mobile browsers refuse the GPU delegate; CPU is slower but works.
      options.baseOptions.delegate = 'CPU';
      landmarker = await vision.PoseLandmarker.createFromOptions(fileset, options);
    }
    state.landmarker = landmarker;
    return landmarker;
  }

  function reset() {
    state.frames = [];
    state.metrics = null;
    state.report = null;
    state.meta = null;
    state.cancelled = false;
    state.finished = false;
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    if (state.stream) {
      for (const track of state.stream.getTracks()) track.stop();
      state.stream = null;
    }
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = '';
    }
    const video = $('video');
    video.pause();
    video.removeAttribute('src');
    video.srcObject = null;
  }

  function drawOverlay(landmarks) {
    const canvas = $('overlay');
    const video = $('video');
    if (!canvas || !video || !landmarks) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = Math.max(2, canvas.width * 0.004);
    for (const side of ['left', 'right']) {
      const idx = M.LANDMARK_INDEX[side];
      const pts = [idx.hip, idx.knee, idx.ankle].map((i) => landmarks[i]);
      if (pts.some((p) => !p)) continue;
      ctx.strokeStyle = 'rgba(15,107,87,0.95)';
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = p.x * canvas.width;
        const y = p.y * canvas.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = 'rgba(15,107,87,0.95)';
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, ctx.lineWidth * 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function sampleFrame(landmarker, video, elapsedMs) {
    let result;
    try {
      result = landmarker.detectForVideo(video, performance.now());
    } catch (err) {
      return;
    }
    const lms = result && result.landmarks && result.landmarks[0];
    if (!lms) return;
    const world = result.worldLandmarks && result.worldLandmarks[0];
    state.frames.push(M.buildFrame(lms, world, video.videoWidth || 720, video.videoHeight || 1280, elapsedMs));
    drawOverlay(lms);
  }

  function progress(pct) {
    const bar = $('progress-bar');
    if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
  }

  /* -------------------------------------------------------------- inputs --- */

  async function startCamera() {
    reset();
    show('screen-analyze');
    status($('analyze-status'), '카메라를 준비하고 있습니다…');
    progress(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      state.stream = stream;
      const video = $('video');
      video.srcObject = stream;
      await video.play();
    } catch (err) {
      show('screen-start');
      status($('start-status'), '카메라를 열 수 없습니다. 브라우저 설정에서 카메라 권한을 확인하거나 영상 파일을 올려 주세요.', 'error');
      return;
    }
    const landmarker = await loadLandmarkerWithUi($('analyze-status'));
    if (!landmarker) return;

    const countdown = $('countdown');
    countdown.hidden = false;
    for (const n of ['3', '2', '1']) {
      countdown.textContent = n;
      status($('analyze-status'), '곧 시작합니다. 전신이 보이게 서 주세요.');
      await new Promise((r) => setTimeout(r, 700));
    }
    countdown.hidden = true;

    const seconds = cfg.sampleSeconds || 20;
    const started = performance.now();
    status($('analyze-status'), '스쿼트를 3회 해 주세요…');
    state.timer = setInterval(() => {
      if (state.cancelled) return;
      const elapsed = performance.now() - started;
      sampleFrame(landmarker, $('video'), elapsed);
      progress((elapsed / (seconds * 1000)) * 100);
      if (elapsed >= seconds * 1000) {
        clearInterval(state.timer);
        state.timer = null;
        finish();
      }
    }, 80);
  }

  async function startFile(file) {
    reset();
    if (!file) return;
    show('screen-analyze');
    status($('analyze-status'), '영상을 불러오고 있습니다…');
    progress(0);
    const video = $('video');
    state.objectUrl = URL.createObjectURL(file);
    video.src = state.objectUrl;
    video.muted = true;
    video.loop = false;
    await new Promise((resolve) => {
      video.onloadedmetadata = resolve;
      video.onerror = resolve;
    });
    if (!video.duration || !Number.isFinite(video.duration)) {
      show('screen-start');
      status($('start-status'), '이 영상 형식을 브라우저에서 열 수 없습니다. 다른 영상을 선택해 주세요.', 'error');
      return;
    }
    if (video.duration > 90) {
      show('screen-start');
      status($('start-status'), '90초가 넘는 영상은 처리할 수 없습니다. 스쿼트 3회 정도만 잘라서 올려 주세요.', 'error');
      return;
    }
    const landmarker = await loadLandmarkerWithUi($('analyze-status'));
    if (!landmarker) return;

    try {
      await video.play();
    } catch (err) {
      show('screen-start');
      status($('start-status'), '영상을 재생할 수 없습니다. 다른 영상을 선택해 주세요.', 'error');
      return;
    }
    status($('analyze-status'), '동작을 계산하고 있습니다…');
    const startedAt = performance.now();
    state.timer = setInterval(() => {
      if (state.cancelled) return;
      const elapsed = performance.now() - startedAt;
      sampleFrame(landmarker, video, video.currentTime * 1000);
      progress((video.currentTime / video.duration) * 100);
      if (video.ended) {
        clearInterval(state.timer);
        state.timer = null;
        finish();
      }
    }, 80);
    video.onended = () => {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
      finish();
    };
  }

  async function loadLandmarkerWithUi(el) {
    status(el, '동작 인식 모델을 불러오고 있습니다…');
    try {
      return await ensureLandmarker();
    } catch (err) {
      show('screen-start');
      status($('start-status'), '동작 인식 모델을 불러오지 못했습니다. 네트워크를 확인한 뒤 다시 시도해 주세요.', 'error');
      return null;
    }
  }

  /* -------------------------------------------------------------- result --- */

  function finish() {
    if (state.finished) return;
    state.finished = true;
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    const metrics = M.computeMetrics(state.frames, { view: state.view });
    state.metrics = metrics;
    stopMedia();
    renderResult(metrics);
    show('screen-result');
  }

  function stopMedia() {
    if (state.stream) {
      for (const track of state.stream.getTracks()) track.stop();
      state.stream = null;
    }
    const video = $('video');
    if (video) video.pause();
  }

  function renderResult(metrics) {
    $('result-meta').textContent =
      metrics.view_label +
      ' · ' +
      (metrics.duration_s || 0) +
      '초 · 분석 프레임 ' +
      metrics.frames_used +
      '/' +
      metrics.frames;
    $('result-chips').innerHTML = V.chips(metrics)
      .map((c) => '<div class="sq-chip"><span>' + V.esc(c.label) + '</span><strong>' + V.esc(c.value) + '</strong></div>')
      .join('');
    $('result-flags').innerHTML = (metrics.flags || [])
      .map((f) => '<li>' + V.esc(f.text) + '</li>')
      .join('');

    const quality = $('quality-block');
    if (metrics.quality.ok) {
      quality.hidden = true;
      quality.innerHTML = '';
      $('teaser-slot').innerHTML = V.teaserHtml(metrics, { priceLabel: cfg.priceLabel });
      const btn = $('teaser-slot').querySelector('[data-sq-open-panel]');
      if (btn) btn.addEventListener('click', openAccess);
    } else {
      $('teaser-slot').innerHTML = '';
      const advice = metrics.quality.reasons.map((r) => '<li>' + V.esc(QUALITY_TEXT[r] || r) + '</li>').join('');
      quality.hidden = false;
      quality.innerHTML =
        '<div class="sq-quality"><h3>이번 영상으로는 리포트를 만들기 어렵습니다</h3>' +
        '<p>아래를 확인하고 한 번만 다시 찍어 주세요.</p><ul>' +
        advice +
        '</ul></div>';
    }
  }

  /* -------------------------------------------------------------- access --- */

  function openAccess() {
    const link = $('buy-link');
    if (cfg.productUrl) {
      link.href = cfg.productUrl;
      link.style.pointerEvents = '';
      link.style.opacity = '';
    } else {
      link.removeAttribute('href');
      link.style.pointerEvents = 'none';
      link.style.opacity = '0.5';
      $('buy-price').textContent = '준비 중';
    }
    $('buy-price').textContent = cfg.productUrl ? cfg.priceLabel : '준비 중';
    const saved = readCode();
    if (saved) $('code-input').value = saved;
    show('screen-access');
    status($('access-status'), '');
  }

  function readCode() {
    try {
      const raw = localStorage.getItem(cfg.codeStorageKey);
      if (!raw) return '';
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.code) return '';
      if (parsed.expires && Date.now() > parsed.expires) {
        localStorage.removeItem(cfg.codeStorageKey);
        return '';
      }
      return parsed.code;
    } catch (err) {
      return '';
    }
  }

  function saveCode(code) {
    try {
      const days = cfg.cacheDays || 30;
      localStorage.setItem(
        cfg.codeStorageKey,
        JSON.stringify({ code, expires: Date.now() + days * 24 * 60 * 60 * 1000 })
      );
    } catch (err) {
      /* storage may be unavailable in private mode; the flow still works */
    }
  }

  function readCachedReport(fingerprint) {
    try {
      const raw = localStorage.getItem(cfg.reportStorageKey);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || parsed.fingerprint !== fingerprint) return null;
      if (parsed.expires && Date.now() > parsed.expires) return null;
      return parsed;
    } catch (err) {
      return null;
    }
  }

  function cacheReport(fingerprint, report, meta) {
    try {
      const days = cfg.cacheDays || 30;
      localStorage.setItem(
        cfg.reportStorageKey,
        JSON.stringify({ fingerprint, report, meta, expires: Date.now() + days * 24 * 60 * 60 * 1000 })
      );
    } catch (err) {
      /* ignore quota errors */
    }
  }

  async function requestReport(code) {
    const metrics = state.metrics;
    if (!metrics) return;
    const cached = readCachedReport(metrics.fingerprint);
    if (cached) {
      renderReport(cached.report, cached.meta);
      return;
    }
    const submit = $('btn-submit');
    submit.disabled = true;
    status($('access-status'), '리포트를 만들고 있습니다. 20~40초 걸립니다…');
    try {
      const res = await fetch(cfg.apiPath || '/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, metrics, goal: state.goal }),
      });
      let data = null;
      try {
        data = await res.json();
      } catch (err) {
        data = null;
      }
      if (!res.ok || !data || !data.ok) {
        const code2 = (data && data.code) || 'upstream_error';
        status($('access-status'), ERROR_TEXT[code2] || ERROR_TEXT.upstream_error, 'error');
        return;
      }
      saveCode(code);
      cacheReport(metrics.fingerprint, data.report, data.meta);
      renderReport(data.report, data.meta);
    } catch (err) {
      status($('access-status'), '네트워크 오류로 리포트를 받지 못했습니다. 잠시 후 다시 시도해 주세요.', 'error');
    } finally {
      submit.disabled = false;
    }
  }

  function renderReport(report, meta) {
    state.report = report;
    state.meta = meta;
    $('report-slot').innerHTML = V.reportHtml(report, meta, state.metrics);
    status($('access-status'), '');
    show('screen-report');
  }

  /* -------------------------------------------------------------- events --- */

  function bind() {
    const camBtn = $('btn-camera');
    if (camBtn) camBtn.addEventListener('click', startCamera);
    $('file-input').addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      e.target.value = '';
      startFile(file);
    });
    $('btn-cancel').addEventListener('click', () => {
      state.cancelled = true;
      reset();
      show('screen-start');
      status($('start-status'), '');
    });
    $('btn-retry').addEventListener('click', () => {
      reset();
      show('screen-start');
    });
    $('btn-close-access').addEventListener('click', () => show('screen-result'));
    $('code-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const code = ($('code-input').value || '').trim();
      if (!code) {
        status($('access-status'), '이용 코드를 입력해 주세요.', 'error');
        return;
      }
      requestReport(code);
    });
    for (const btn of document.querySelectorAll('[data-goal]')) {
      btn.addEventListener('click', () => {
        state.goal = btn.getAttribute('data-goal');
        for (const other of document.querySelectorAll('[data-goal]')) other.classList.toggle('is-on', other === btn);
      });
    }
    for (const btn of document.querySelectorAll('[data-view]')) {
      btn.addEventListener('click', () => {
        state.view = btn.getAttribute('data-view');
        for (const other of document.querySelectorAll('[data-view]')) other.classList.toggle('is-on', other === btn);
      });
    }
    $('btn-print').addEventListener('click', () => window.print());
    $('btn-new').addEventListener('click', () => {
      reset();
      show('screen-start');
    });
    window.addEventListener('pagehide', reset);
  }

  // Layout review without a camera, a video or a key: ?reportPreview=1
  function renderPreview() {
    const fixtures = V.previewFixtures();
    state.metrics = fixtures.metrics;
    state.view = fixtures.metrics.view;
    renderResult(fixtures.metrics);
    $('report-slot').innerHTML = V.reportHtml(
      fixtures.report,
      { generatedAt: new Date().toISOString(), model: 'gpt-6.1-terra', tokens: { input: 2410, output: 1980 } },
      fixtures.metrics
    );
    show('screen-result');
  }

  function init() {
    bind();
    if (/[?&]reportPreview=1/.test(location.search)) {
      renderPreview();
      return;
    }
    if (!window.isSecureContext) {
      status($('start-status'), '카메라는 HTTPS 또는 localhost에서만 열립니다. 영상 파일을 올려 주세요.', 'error');
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
