---
phase: 7
title: "Automatic Q&A Pipeline"
status: paused
priority: P1
effort: "6-10 team-days plus Kaggle model gates"
dependencies: [2, 3, 4]
optional_dependencies: [6]
---

# Phase 7: Automatic Q&A Pipeline

## Context links

- Competition contract: `../../docs/competition-requirements.md`
- Evaluation protocol: `../../docs/evaluation-protocol.md`
- Kaggle runtime: `../../docs/kaggle-runbook.md`
- Research: `../reports/research-260808-2105-automatic-qa-request-2-report.md`

## Overview

**Trạng thái hiện tại:** Phase 7A contracts/query routing đã implement và verify local. Phase 7B–7C tạm dừng theo quyết định ưu tiên Request 1; chỉ tiếp tục sau Request 1 B0–B3 integration/review và locked private Kaggle promotion gate.

Mở rộng Request 1 retriever thành Request 2 tự động end-to-end. Người dùng chỉ nhập raw prompt chứa mô tả sự kiện và câu hỏi. Pipeline tự tìm video/khoảnh khắc, tự sinh answer, tự verify/rank và xuất `1–100` triples `<video_id, frame_id, answer>`. Không có bước chọn frame, nhập/sửa answer hoặc chỉnh rank bởi người dùng trong critical path.

Baseline kỹ thuật dùng deterministic query routing, multilingual CLIP retrieval, exact-frame temporal evidence, pluggable video VLM answer engine và strict verifier. OCR/ASR tự động là gated enrichers; không yêu cầu human fallback. Nếu mọi hypothesis fail, pipeline trả empty response list cùng aggregate failure status thay vì hallucinate hoặc hỏi người dùng.

## Requirements

### Functional

- Nhận đúng một raw prompt; không cần task-specific field do người dùng điền thêm.
- Tự tách event/localization phrase và question bằng deterministic bilingual parser có full-query fallback.
- Retrieve wide keyframe pool, group thành bounded temporal evidence windows và sample exact original frame IDs.
- Video VLM tự sinh short answer và chọn evidence frame slot trong sampled window.
- Hard validator và automatic verifier reject ungrounded/malformed hypotheses.
- Optional OCR/ASR evidence chạy tự động theo bounded router, không yêu cầu operator.
- Joint-rank và export `1–100` canonical `QaResponse` records.
- Hỗ trợ answer Việt/Anh; raw answer không bị normalization mutate.

### Non-functional

- Deterministic với cùng prompt/index/model/config; generation dùng `do_sample=False`.
- Mỗi response truy ngược được tới video, exact original frame, window, model revision và evidence path nội bộ.
- Không `trust_remote_code=True`, cloud/commercial API hoặc network dependency trong frozen runtime.
- Dataset, model cache, frame tensors và private labels chỉ ở Kaggle/private runtime.
- Frame/window/token/candidate budgets bounded; failures fail closed, không OOM toàn query.
- Aggregate report không chứa raw prompt, frames, generated reasoning, OCR/ASR content hoặc ground truth.

## Architecture

```text
raw prompt
  │
  ▼
deterministic QA parser/router
  ├── raw localization variant
  ├── event-description variant
  └── answer type / modality hints
  │
  ▼
multilingual text encoder → exact CLIP search → video/time clusters
  │
  ▼
coarse evidence windows → deterministic exact-frame samples
  │
  ├── video VLM answer engine ────────────────┐
  ├── optional OCR on sampled evidence frames ├─► answer hypotheses
  └── optional ASR segments overlapping window┘
  │
  ▼
hard provenance validation → fixed-label support verifier
  │
  ▼
coarse→refine evidence pass → joint ranking/diversity
  │
  ▼
1–100 <video_id, original_frame_id, raw_answer>
```

### Component boundaries

1. Retrieval localizes evidence; không sinh answer.
2. Evidence builder owns exact frame IDs; model chỉ chọn bounded frame slot.
3. Answer engine consumes frames/question and returns one bounded protocol line.
4. Hard verifier bắt buộc kiểm tra schema/provenance; learned support verifier dùng cùng VLM chỉ là ablation vì không độc lập và tăng latency.
5. Ranker assembles canonical responses từ components đã promote; evaluator remains independent.

## Contracts

```python
@dataclass(frozen=True, slots=True)
class QaQuery:
    raw_text: str
    event_description: str
    question: str
    answer_type: str
    use_ocr: bool
    use_asr: bool

@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    video_id: str
    start_frame_id: int
    end_frame_id: int
    seed_frame_ids: tuple[int, ...]
    sampled_frame_ids: tuple[int, ...]
    retrieval_score: float
    raw_ranks: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class FrameEvidence:
    slot: int
    frame_id: int
    image: object

@dataclass(frozen=True, slots=True)
class AnswerHypothesis:
    video_id: str
    frame_id: int
    raw_answer: str
    normalized_answer: str
    retrieval_score: float
    support_score: float
    agreement_score: float
    joint_score: float

@dataclass(frozen=True, slots=True)
class QaPipelineStatus:
    query_parsed: bool
    retrieved_windows: int
    answered_windows: int
    valid_hypotheses: int
    response_count: int
    vlm_circuit_open: bool
    elapsed_ms: float
```

`image` remains in memory and is never serialized by `to_dict()`/reports.

## Related code files

### Create

- `src/aic_retrieval/qa.py` — query/contracts/window/ranking orchestration.
- `src/aic_retrieval/qa_models.py` — answer-engine/verifier protocols and lazy model adapter.
- `src/aic_retrieval/qa_evidence.py` — exact-frame sampling and optional evidence interfaces.
- `scripts/answer_query.py` — prompt-only automatic CLI.
- `scripts/benchmark_qa.py` — locked Request 2 benchmark.
- `tests/test_qa_pipeline.py` — pure contracts/mock end-to-end tests, including no-human public API signatures.
- `tests/test_qa_models.py` — mocked processor/model output and failure tests.
- `config/qa-baseline.yaml` — automatic pipeline budgets/weights/model identity.
- `config/qa-benchmark.example.yaml` — private query-set paths and output.
- `notebooks/aic-qa-model-gate-kaggle.ipynb` — public synthetic T4 candidate gate.
- `notebooks/aic-qa-kaggle.ipynb` — frozen automatic Request 2 run.

### Modify

- `src/aic_retrieval/submission.py` — write/validate automatic Q&A response artifact if final transport requires it.
- `src/aic_retrieval/experiment.py` — Q&A stage/resource metrics only if existing generic record cannot represent them.
- `scripts/prepare_kaggle_bundle.py` — include new source/config/tests automatically; no model/data weights.
- `tests/test_kaggle_bundle.py` — required Q&A files and forbidden artifact coverage.
- `docs/kaggle-runbook.md` — model gate, automatic Q&A execution, privacy/resources.
- `pyproject.toml` — optional extras only after model gate selects compatible dependencies.

### Keep unchanged unless contract evidence requires change

- `src/aic_retrieval/evaluation.py` canonical Q&A scoring.
- `docs/evaluation-protocol.md` official-formula boundary.
- Request 1 baseline notebook/config.

## Implementation phases

### 7A. Automatic contracts and query routing

1. Add `QaError` and frozen contracts with strict type/range/finite validation.
2. Implement `parse_qa_query(raw_text)`:
   - preserve raw text;
   - split explicit `Câu hỏi:`/`Question:` markers;
   - otherwise split final interrogative clause only when deterministic;
   - fallback full raw prompt for both event and question;
   - reject only empty/control-character input.
3. Add bounded bilingual answer-type router: count, visible text/name/number, speech, color, person, location, object/action, unknown.
4. Router sets OCR/ASR hints; always keeps VLM and original prompt.
5. Add tests for Vietnamese/English, punctuation, no marker, negation, malformed/empty input and no raw mutation.
6. Add introspection/CLI tests proving public Request 2 entry points accept no user answer, frame selection, rank override or correction callback.

### 7B. Retrieval and temporal evidence windows

1. Encode at most two localization variants: raw prompt and distinct event description.
2. Reuse `QueryEncoderRuntime` and exact index; preserve per-variant raw ranks/scores.
3. Merge variants deterministically using one configured strategy; benchmark RRF vs normalized mean/max before promotion.
4. Group hits by video and temporal seconds using manifest FPS; keep multiple clusters/video.
5. Build validated `EvidenceWindow` records bounded by video frame count.
6. Implement deterministic sampler:
   - include top seed IDs;
   - uniform ordered frames across window;
   - add bounded near-seed frames;
   - no duplicates, exact original frame IDs;
   - decode BGR→RGB explicitly;
   - never persist frame arrays.
7. Define timestamp/frame conversion once: original `frame_id` is zero-based; `time_s = frame_id / fps`; ASR/window seconds use explicit floor/ceil plus video-bound clamp.
8. Implement coarse→refine windows; refined window centers on valid evidence slot/best seed.
9. Add synthetic tests for bounds, variable/non-integer FPS, exact timestamp boundaries, duplicate seeds, decoder failure and deterministic output.

### 7C. Kaggle T4 VLM candidate gate

Evaluate separately; do not install/load all candidates in production runtime.

1. Primary: `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`.
2. Control: `HuggingFaceTB/SmolVLM2-2.2B-Instruct@482adb537c021c86670beed01cd58990d01e72e4`.
3. Low-resource control: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf@74dd0bf867a4cda7950c17663794267c60cf4b40`.
4. Load built-in Transformers classes with exact revision, `trust_remote_code=False`, T4-safe FP16, SDPA, `do_sample=False`.
5. Use same bounded public synthetic frame/resolution/output budget for every candidate.
6. Test Vietnamese/English: count, action, temporal order, visible text, no-evidence and evidence-slot protocol.
7. Measure startup, preprocess, generation p50/p95, peak allocated/reserved VRAM, host RAM and protocol/provenance pass rate.
8. Audit exact checkpoint license file, model-card metadata, built-in class support and availability from a stable pinned Transformers wheel; development-source-only compatibility cannot enter frozen runtime.
9. Reject model on load/revision/license/VRAM/parser/reliability failure. Model-card scores cannot promote.
10. Select one winner; pin compatible Transformers/dependency versions only then.

T4 acceptance threshold must be registered before run. Initial operational target: leave GPU headroom for text encoder/processor and avoid repeated OOM; exact numeric ceiling comes from measured T4 baseline, not model-card claims.

### 7D. VLM answer engine and strict protocol

1. Define `AnswerEngine` protocol and mock implementation before real adapter.
2. Real adapter lazy-loads one selected VLM and processor once per session.
3. Supply ordered sampled frames with slot labels plus raw question/event description.
4. Request exactly one line:

```text
<frame_slot>\t<short_answer>
```

5. Parser requires exactly one line/tab, slot integer in range, non-empty bounded answer, no controls/extra prose.
6. Map slot to project-owned exact frame ID; model cannot emit arbitrary ID.
7. Discard generated content after parse; expose only bounded counts/timing/status.
8. On per-window model failure, drop window and continue. Repeated load/OOM/runtime failure opens in-memory circuit for query/session; no user fallback.
9. Empty final hypotheses return empty canonical responses with aggregate status, not fabricated answer.

### 7E. Automatic verifier and joint ranking

1. Hard-validate manifest/video/window/frame/answer identity; đây là mandatory verifier baseline.
2. Compute cross-window agreement from genuinely different coarse/refined or adjacent views.
3. Treat fixed-label conditional-likelihood `SUPPORTED` vs `UNSUPPORTED` từ cùng VLM như optional ablation, không như ground-truth verifier; measure marginal score và thêm latency.
4. Normalize finite score components using training distribution.
5. Joint score baseline combines retrieval + hard-valid evidence + agreement; optional support/OCR/ASR contributions chỉ bật sau promotion và không cứu weak video.
6. Expand verified answer across a bounded set of strong frames inside refined window to improve interval recall.
7. Deduplicate by normalized `(video_id, frame_id, answer)` while preserving raw answer.
8. Apply configurable video/window diversity and cap at 100.
9. Add wrong-video/right-answer, right-video/wrong-frame, malformed answer, verifier conflict and tie-order tests.
10. Add counting-specific invariants: same person across adjacent frames must not be summed per frame; count answers require one window-level hypothesis, cross-view agreement and a dedicated repeated-identity failure set before promotion.

### 7F. OCR and ASR gated enrichers

1. Define nullable `OcrEvidenceProvider` and `AsrEvidenceProvider`; no dependency in core tests.
2. VLM internal OCR remains baseline.
3. OCR experiment:
   - PP-OCRv5 mobile/Latin on sampled top-window frames/crops;
   - trigger only text/name/sign/score categories;
   - preserve text/confidence/frame provenance in private memory/artifact;
   - verify Vietnamese diacritics empirically.
4. ASR experiment:
   - `openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d`;
   - precompute timestamped segments as resumable Kaggle artifact;
   - query-time overlap lookup only;
   - map timestamps to frame intervals via canonical FPS;
   - selected submission frame still comes from visual evidence.
5. Missing OCR/ASR continues VLM-only; no user interaction.
6. Promote each provider only on its relevant held-out subset plus whole-set marginal gain/resource gate.

### 7G. Prompt-only CLI and Kaggle notebook

1. `scripts/answer_query.py` requires prompt, index, dataset/manifest, encoder config and QA config; no answer/frame flags.
2. Runtime loads index/text encoder/VLM once, then exposes `answer_query(text)` for repeated prompts.
3. Output private artifact contains canonical responses and allowed aggregate provenance/status only.
4. Assert raw prompt, generated reasoning, frame pixels, OCR/ASR text and private labels absent from aggregate summary.
5. Kaggle notebook verifies accelerator/model/source/index/manifest revisions before query.
6. `Save Version → Save & Run All` must complete without manual cell edits or answer correction.
7. No automatic external submission; transport adapter remains separate until BTC publishes protocol.

### 7H. Benchmark and promotion

1. Define private Q&A query set with raw prompt, GT video/interval, accepted answers, language and answer-type tags.
2. Lock train/held-out IDs before model/config tuning.
3. Report stage metrics:
   - correct-video/window recall;
   - localization accuracy;
   - answer proxy conditional on correct localization;
   - joint Q&A R@1/R@5/R@20/R@50/R@100 and Final Score;
   - output parse/provenance/empty-response rates;
   - per-stage and full p50/p95, peak VRAM/RAM, model/index size.
4. Required ablations:
   - raw-only vs raw+event retrieval;
   - single-scale vs coarse→refine;
   - each VLM candidate;
   - winner with hard provenance only vs optional same-VLM support scorer;
   - OCR/ASR only on preregistered relevant subsets;
   - winner VLM-only vs promoted automatic enrichers.
5. Keep public synthetic model-gate fixtures, training split and held-out split disjoint; record every model/config decision made after seeing each split. Never tune on held-out or Kaggle output examples.
6. Promote only if held-out joint score improves, R@1/R@5 guardrails pass, automatic provenance passes, resources fit T4 and exact identities reproduce.
7. Every rejected candidate/modality gets explicit decision report; default config enables only promoted components.

## Tests and validation

### Local gates

```bash
python -m unittest tests.test_qa_pipeline tests.test_qa_models tests.test_retrieval tests.test_evaluation tests.test_kaggle_bundle -v
python -m unittest discover -s tests -v
python -m compileall src scripts
git diff --check
```

Also parse every Python file and notebook code cell with Python 3.10-compatible `ast.parse`.

Local tests mock model/processor and use tiny videos/frames; no Hugging Face download, private labels or full dataset.

### Kaggle gates

- NVIDIA T4 exact device verified.
- Model/processor `_commit_hash` equals pinned revision when runtime exposes it.
- No remote code; no Internet required after cache/attached model artifact prepared.
- Public synthetic model gate passes before private benchmark.
- Frozen Request 2 notebook accepts prompt only and emits canonical responses automatically.
- Aggregate summary passes privacy assertions.

## Success criteria

- [ ] Input contract exposes only raw prompt plus system/config paths; no user answer/frame/rank controls or correction callbacks.
- [ ] Query parser always produces deterministic localization/question fallback without model call.
- [ ] Correct event windows use canonical videos and exact original frame IDs.
- [ ] Model outputs only bounded slot + short answer protocol; parser/provenance failures cannot enter responses.
- [ ] Pipeline creates `1–100` responses for successful queries; all-failure path returns empty list/status without hallucination.
- [ ] Every response has model/index/config/evidence provenance internally.
- [ ] Raw answer preserved; normalized answer only for dedup/matching.
- [ ] Wrong video, wrong frame or wrong answer score zero in evaluator tests.
- [ ] Fresh T4 candidate gate selects/rejects models using measured quality/reliability/resources plus exact license and stable Transformers wheel audit.
- [ ] OCR/ASR remain automatic, optional and individually promotable.
- [ ] Locked held-out report separates retrieval, localization, answer and joint metrics.
- [ ] End-to-end benchmark contains no human edits or selection.
- [ ] Frozen notebook runs `Save Version → Save & Run All` from prompt to ranked output.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| VLM hallucinates plausible answer | strict slot protocol, mandatory hard provenance, cross-view agreement; same-VLM support scorer chỉ bật nếu ablation thắng |
| VLM picks correct answer but wrong frame | project-owned sampled slots, refined window, joint localization gate |
| Small VLM weak Vietnamese | multilingual retrieval + same labeled bilingual model gate; compare candidates, do not assume |
| Counting across time double-counts people | ordered frames, coarse→refine, cross-view agreement, count-specific held-out subset |
| OCR misses Vietnamese diacritics | VLM baseline; separate Latin PP-OCR experiment with empirical Vietnamese gate |
| Audio answer absent visually | timestamped Whisper artifact, bounded overlap lookup, visual exact frame remains required |
| T4 OOM/slow | 2B-first candidates, bounded frames/resolution/tokens, one model/session, measured gate |
| Free-form model output malformed | one-line tab protocol, no JSON reliance, reject/continue |
| Semantic matcher unknown | preserve short raw answers; exact and labeled local proxy remain separate |
| No labeled Request 2 set | create/lock private set before tuning; do not claim quality from contact sheets |
| Optional tool missing | VLM-only continues automatically; no human fallback |

## Rollback

- Disable OCR/ASR independently and return to VLM-only automatic pipeline.
- If verifier hurts held-out score, disable verifier weighting while retaining hard provenance validation.
- If no VLM passes T4 automatic gate, mark Request 2 blocked by model/runtime evidence; do not replace answer generation with manual input.
- Request 1 exact baseline and canonical evaluator remain unchanged.
