/* Site configuration for the squat report.
 *
 * Everything a launch decision changes is here; no rebuild step needed.
 * productUrl: the marketplace product page for the 리포트 이용권. While it is
 *   empty the app shows "구매 링크 준비 중" and still accepts a code, so the
 *   paid path can be tested before the product is published.
 */
window.SQUAT_REPORT_CONFIG = {
  priceLabel: '4,900원',
  productUrl: '',
  apiPath: '/api/report',
  cacheDays: 30,
  codeStorageKey: 'sq_report_code',
  reportStorageKey: 'sq_report_cache',
  sampleSeconds: 20,
  poseModel: 'full',
  modelUrls: {
    full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  },
  // Pinned versions, checked 2026-11-18: the bundle exports FilesetResolver,
  // PoseLandmarker, createFromOptions and forVisionTasks, and the wasm folder
  // holds vision_wasm_internal.wasm.
  wasmUrl: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm',
  visionUrl: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3',
};
