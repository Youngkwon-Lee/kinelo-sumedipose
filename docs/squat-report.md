# Squat report — paid AI report layer (product note)

A small web app in `web/` plus one serverless function in `api/report.js` that
turns a squat video into a numeric summary and, with a paid access code, a
4-week training report.

This directory is **not** part of the research pipeline. The research claims,
data policy and interpretation caveats in `README.md`, `DATA_POLICY.md` and
`docs/*.md` are unchanged and none of them are softened by the product: the app
repeats the same limits and never presents its numbers as validated
measurement or as a clinical result.

## 1. What the buyer sees

| | Free (no code) | Paid (access code) |
| --- | --- | --- |
| Repetitions counted | ● | ● |
| Knee angle at the bottom (3D estimate + 2D projection) | ● | ● |
| Left/right difference, tempo, trunk lean, stance | ● | ● |
| Plain-language interpretation (why it looks like this) | – | ● 3 paragraphs |
| Priority list of what to fix first | – | ● |
| **4-week routine** — 3 sessions/week, 2–3 steps each, sets/reps/cues | – | ● |
| How-to for each exercise (18-exercise catalog) | – | ● |
| Checkpoints and "measure again on week 4" | – | ● |
| PDF (browser print) | – | ● |

Nothing is taken away from a free visitor: the whole numeric result is free,
and the paid part is material that did not exist before.

## 2. Where the numbers come from

```
video frame → MediaPipe PoseLandmarker (browser, CDN)
            → web/squat-metrics.js  buildFrame()  → hip/knee/ankle angles
            → computeMetrics()                   → reps, depth, asymmetry, tempo, trunk, stance
            → POST /api/report  (numbers only)   → OpenAI structured output
            → web/report-view.js                 → report + PDF
```

- `includedAngle()` is a port of `src/sumedipose/geometry.py:joint_angle`.
- Landmark indices are the ones `src/sumedipose/benchmark.py` maps for its pose
  baseline: hip 23/24, knee 25/26, ankle 27/28.
- `tests/test_web_metrics_parity.py` compares the JavaScript angle with the
  Python function on 200 random triples (1e-6°) and checks the index mapping
  against `benchmark.JOINT_NAMES`, so the two implementations cannot drift.
- Both the 2D projection angle and the model's 3D world-landmark angle are
  computed and shown; the repository already documents why they must not be
  ranked against each other.

Reps are found on the knee-angle signal with hysteresis (enter at 55 % and exit
at 78 % of the observed range), widened to the last/first standing frame, and an
unfinished final rep is dropped rather than guessed.

## 3. API contract

`POST /api/report`

```json
{
  "code": "SQ-XXXX-XXXX",
  "goal": "strength | mobility | consistency",
  "metrics": { "rep_count": 3, "depth": {...}, "asymmetry": {...}, "tempo": {...},
               "trunk": {...}, "stance": {...}, "consistency": {...},
               "frames": 246, "frames_used": 231, "duration_s": 11.8, "range_deg": 74.3 }
}
```

→ `200 { ok, report: { headline, summary_line, sections[3], findings[], weeks[4], checkpoints[], cautions[], disclaimer }, meta: { model, tokens, cost_usd, cost_krw, generatedAt } }`

Errors: `403 invalid_code`, `400 insufficient_measurement`, `413 body_too_large`,
`429 rate_limited`, `502 upstream_error|invalid_output|unsafe_output|refused`,
`503 not_configured`, `405 method_not_allowed`.

### What the function refuses to do

- No video, image, landmark, name or contact detail is accepted; the body is
  sanitised to a fixed numeric whitelist with physical clamps.
- No key, no code list and no prompt ever appears in a response (tested).
- Reports are rejected if they carry a medical claim (`진단·질환·파열·염증·처방·
  치료법·디스크`), if a week is missing, or if fewer than two sections survive.
  Self-negating copy such as "의료 진단이 아닙니다" is not treated as a claim.
- Exercise ids are validated against the catalog, and the model may only use
  those ids.
- The disclaimer is fixed server copy and overwrites whatever the model wrote.
- Rate limits: 6/hour and 15/day per IP, 300/day per instance. No payload is
  logged or stored; usage counters are in-memory and hold no buyer data.

## 4. Configuration

| Name | Where | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | Vercel env | OpenAI project key (`kinelo-pilot`). |
| `SQ_REPORT_ACCESS_CODES` | Vercel env | Comma-separated access codes, e.g. `SQ-7K2M-2411`. |
| `SQ_REPORT_MODEL` | Vercel env, optional | Default `gpt-6.1-terra`; `gpt-6-luna` cuts cost ~10×. |
| `productUrl` | `web/config.js` | Marketplace product page. Empty shows "구매 링크 준비 중" and still accepts a code. |
| `priceLabel` | `web/config.js` | Shown on the paid card; must match the product page. |

`vercel.json` serves `web/` as the static output and `api/report.js` as a
function with `maxDuration: 60`. If the Vercel project is created with its Root
Directory pointed at `web/`, the function must instead live inside that root —
keep the Root Directory at the repository root.

## 5. Cost and margin (planning estimate, not a measured call)

Prompt sizes measured from the code: system 1,312 characters, user 1,235,
JSON schema 1,903 → **≈2,200 input tokens**; output capped at 6,000 tokens,
expected **≈2,200** for 4 weeks × 3 sessions.

| Model | Price ($/1M in / out) | Est. cost / report | Note |
| --- | --- | --- | --- |
| `gpt-6.1-terra` (default) | 1.00 / 5.00 | **≈19원** | quality first |
| `gpt-6.1-sol` | 2.00 / 10.00 | ≈38원 | |
| `gpt-5.4-mini` | 0.75 / 4.50 | ≈17원 | |
| `gpt-6-luna` | 0.10 / 0.50 | **≈2원** | cost first |

Marketplace fee (래피드 Free: 1.6 % + PG 2.9 % + VAT on fees) = **4.95 %**:

| Price | Fee | AI cost | Net | Margin |
| --- | --- | --- | --- | --- |
| 3,900원 | 193원 | 19원 | 3,688원 | 94.6 % |
| 4,900원 | 243원 | 19원 | 4,638원 | 94.7 % |
| **5,900원** | 292원 | 19원 | 5,589원 | 94.7 % |
| 6,900원 | 342원 | 19원 | 6,539원 | 94.8 % |

AI cost is not the constraint at any of these prices: a 4-week routine on one
call is roughly 1/300 of the sale price. The constraints are traffic and
conversion, and the honesty limit below.

## 6. Verification

| Check | Result |
| --- | --- |
| Existing research tests | 135 passed, unchanged |
| New Python tests (`tests/test_web_metrics_parity.py`) | parity + benchmark index mapping + runs the whole Node suite |
| Node suite (`tests/web/*.test.cjs`) | metrics (10), catalog (4), API (22), view (6), page controller (2) — 44 tests |
| API behaviour | exercised with a mocked OpenAI response: 200 / 403 / 400 / 405 / 413 / 429 / 502 / 503 paths |
| Report layout | `drafts` sample PDFs built from the real renderer |
| **Not verified here** | live camera or video inference in a real browser (no camera/video in the build environment), a real OpenAI call (no key), and a Vercel preview (no linked project) |

The browser-side pose pipeline is therefore code-complete and unit-tested at the
frame/metric level, but it has **not** been run against a real person's video
yet. First job after deploy is one real recording, one real code, one real call.

## 7. Boundary this product must keep

The repository's own position is that these metrics are exploratory: no
Vicon-registered accuracy, no exposure-time synchronisation, no clinical
validity. The product says the same thing in three places — the start screen,
the report's "측정 한계" block and the fixed disclaimer — and the report never
claims to diagnose or to grade a movement as normal or abnormal.

If the wording ever starts implying medical assessment (for example "무릎 질환
위험"), the report must stop: that is a different product with a different
regulatory position.
