---
title: "AIC 2026 Request 2 Automatic Q&A Research"
date: "2026-08-08"
status: completed
scope: "Kaggle NVIDIA T4, prompt-only input, automatic ranked Q&A output"
---

# AIC 2026 Request 2 Automatic Q&A Research

## Executive summary

Request 2 phải tự động end-to-end. Người dùng chỉ nhập một prompt chứa mô tả sự kiện và câu hỏi. Hệ thống tự trả `1–100` response `<video_id, frame_id, answer>`. Không có bước chọn frame, nhập answer, sửa answer hoặc đổi rank bởi người dùng trong critical path.

Kiến trúc đề xuất:

```text
raw prompt
  → deterministic query decomposition + answer-type routing
  → multilingual CLIP event retrieval trên exact NumPy index
  → temporal clustering + bounded multi-scale evidence windows
  → deterministic exact-frame sampling
  → video VLM sinh answer + evidence slot
  → automatic evidence verifier + optional OCR/ASR support
  → joint ranking + frame expansion + dedup
  → strict canonical Q&A responses
```

Không khóa model trước empirical gate. Candidate đầu tiên nên smoke-test là `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`: Apache-2.0 theo Hugging Face metadata, built-in Transformers, 2B, video input và temporal grounding. Control là `HuggingFaceTB/SmolVLM2-2.2B-Instruct@482adb537c021c86670beed01cd58990d01e72e4`: Apache-2.0, card báo khoảng 5,2 GB GPU RAM cho video nhưng English-only. Low-resource control là `llava-hf/llava-onevision-qwen2-0.5b-ov-hf@74dd0bf867a4cda7950c17663794267c60cf4b40`, Apache-2.0, English/Chinese. Chỉ promote model thắng locked Vietnamese/English AIC-style joint-Q&A benchmark trên T4.

`Qwen/Qwen2.5-VL-3B-Instruct` không được chọn: checkpoint hiện dùng Qwen Research License, trong khi Qwen3-VL-2B có Apache-2.0 metadata và mới hơn. Text-only Qwen planner đã reject không được tái dùng; đó là component/model khác với video VLM candidate.

## Non-negotiable contract

### Input

Một raw prompt duy nhất, ví dụ:

```text
Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?
```

### Output

```text
video_xyz,3450,5
```

Hoặc canonical Python record:

```python
QaResponse(video_id="video_xyz", frame_id=3450, answer="5")
```

### Invariants

- Không human-in-the-loop trong query runtime.
- Mọi answer gắn một video và exact original frame ID.
- Answer có thể Việt hoặc Anh; raw answer không bị normalization mutate.
- Tối đa 100 responses, ranked tốt nhất trước.
- Malformed model output không được tự sửa bằng dữ liệu bịa.
- Model/API Internet không thuộc runtime bắt buộc.
- Frames, raw prompt, generated reasoning, OCR/ASR snippets và private labels không vào aggregate report/source bundle.

## Current reusable project assets

Đã có:

- canonical manifest và exact original-frame mapping;
- exact NumPy inner-product index: 177.321 rows, 512 dimensions;
- `QueryEncoderRuntime` cho OpenAI CLIP control và multilingual CLIP candidate;
- wide retrieval, temporal dedup, video cap và top-100 validation;
- lazy exact-frame decode/window iterator;
- canonical `QaGroundTruth`, `QaResponse`, Q&A scorer và submission loader;
- deterministic experiment/report infrastructure;
- source-only Kaggle bundle builder.

Thiếu:

- automatic Q&A query contract/parser;
- evidence-window builder gắn video metadata/FPS;
- deterministic frame sampler;
- answer-engine interface và real VLM adapter;
- automatic verifier/joint ranker;
- OCR/ASR evidence adapters;
- Q&A CLI/notebook/benchmark;
- labeled Request 2 dev/held-out set.

## Recommended architecture

```text
                               ┌─ full raw prompt ───────────────┐
raw prompt ─► QA parser/router ├─ event/localization phrase ────┼─► text embeddings
                               └─ answer-type metadata ──────────┘
                                                                  │
exact CLIP index ◄─────────────────────────────────────────────────┘
      │
      ▼
wide keyframe candidates
      │
      ▼
video/time clustering ─► coarse windows ─► exact sampled frame IDs
                                               │
                               ┌───────────────┼───────────────┐
                               ▼               ▼               ▼
                            video VLM      OCR evidence     ASR evidence
                               │               │               │
                               └──── answer hypotheses + provenance ────┐
                                                                       ▼
                                                     hard validation + verifier
                                                                       │
                                                                       ▼
                                                       joint score + diversity
                                                                       │
                                                                       ▼
                                                     1–100 QaResponse records
```

### Separation of concerns

1. Retrieval localizes likely event windows; không sinh answer.
2. Evidence builder owns exact frame IDs; model không tự invent frame ID.
3. Answer engine sinh short answer và chọn một slot từ sampled frames.
4. Verifier đánh giá evidence support; không dùng self-reported confidence.
5. Ranker kết hợp retrieval/evidence/model scores; evaluator giữ official-formula boundary.

## Query decomposition without a planner LLM

Không dùng generative planner cho decomposition. Parser deterministic:

1. Preserve `raw_text` byte-for-byte/Unicode.
2. Nếu prompt có `Câu hỏi:`/`Question:`, dùng marker.
3. Nếu không, lấy interrogative clause cuối làm `question`; prefix làm `event_description` khi split rõ.
4. Nếu split không chắc, dùng full raw prompt cho cả localization và question; không fail.
5. Tạo tối đa hai retrieval variants: raw prompt và event description distinct/non-empty.
6. Detect answer type bằng bounded bilingual rules:
   - count: `bao nhiêu`, `mấy`, `how many`;
   - visible text/name/number: `ghi gì`, `tên gì`, `biển`, `what text`, `what name`;
   - audio/speech: `nói gì`, `phát biểu`, `gọi tên`, `said`, `announced`;
   - time/color/person/location/object/action categories.
7. Router chỉ chọn evidence tools/prompt template. Nó không bỏ original query và không quyết định answer.

Fallback này tránh parser failure làm mất query và tránh latency/instability của text planner.

## Event retrieval

### Baseline

- Encode localization variants bằng pinned multilingual CLIP candidate.
- Search exact index với configurable wide depth; giữ raw ranks/scores.
- Union variants rồi aggregate deterministic; RRF hoặc normalized max/mean phải đăng ký trước benchmark.
- Group hits theo video và temporal distance tính bằng seconds từ FPS, không hard-code frame distance.
- Giữ nhiều windows/video vì đúng event có thể ở vùng khác.

### Candidate window record

```python
@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    video_id: str
    start_frame_id: int
    end_frame_id: int
    seed_frame_ids: tuple[int, ...]
    sampled_frame_ids: tuple[int, ...]
    retrieval_score: float
    raw_ranks: tuple[int, ...]
```

Validation:

- IDs non-negative, ordered, inside source video bounds;
- sampled IDs thuộc window;
- seed/sample không duplicate;
- video tồn tại trong manifest;
- all scores finite.

### Multi-scale strategy

Không gửi full video vào VLM. Với mỗi top cluster:

1. Coarse window quanh seed/cluster, bounded seconds.
2. Sample uniform frames + exact top-hit frames.
3. VLM đánh giá/answer coarse window.
4. Refine quanh evidence slot hoặc best seed bằng denser exact frames.
5. Answer/verify lại trên refined window.

Starting values chỉ là smoke config, không promotion defaults:

```text
candidate_depth: 500
max_candidate_videos: 20
max_coarse_windows: 16
coarse_frames_per_window: 12
max_answer_windows: 8
refine_frames_per_window: 16
```

Tune bằng locked train split; report sensitivity. Visual frame budget phải bounded để T4 không OOM.

## Deterministic exact-frame sampling

Project tự decode frames thay vì giao raw video path cho model sampler:

- map every tensor/image back to exact original `frame_id`;
- sample uniform endpoints/interior;
- always include top retrieval seed IDs;
- add near-seed frames within bounded radius when budget remains;
- convert BGR→RGB explicitly;
- no frame arrays written to disk/report by default;
- sampling result deterministic for same window/config.

VLM receives ordered frame tensor/list plus slot labels. Model outputs a slot, never arbitrary frame ID.

## Video VLM model gate

### Candidate A — primary research candidate

`Qwen/Qwen3-VL-2B-Instruct`

- revision: `89644892e4d85e24eaac8bacfd4f463576704203`;
- Hugging Face metadata: Apache-2.0, Transformers;
- built-in `Qwen3VLForConditionalGeneration`/`AutoProcessor`;
- official Transformers processor supports videos and `sample_frames(..., num_frames=..., fps=...)`;
- model card claims hour-scale video and second-level temporal indexing;
- no `trust_remote_code=True` requirement documented;
- exact T4 VRAM/latency and Vietnamese quality: unverified, mandatory smoke gate.

Why first: smallest current Qwen3-VL dense instruct checkpoint, Apache metadata, temporal-video architecture, built-in Transformers.

### Candidate B — memory-known control

`HuggingFaceTB/SmolVLM2-2.2B-Instruct`

- revision: `482adb537c021c86670beed01cd58990d01e72e4`;
- Apache-2.0;
- built-in `AutoModelForImageTextToText`/`AutoProcessor`;
- model card reports 5,2 GB GPU RAM for video;
- Video-MME 52.1, MLVU 55.2, MVBench 46.27 reported by card;
- English listed; Vietnamese query quality unverified and likely main risk.

### Candidate C — low-resource control

`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`

- revision: `74dd0bf867a4cda7950c17663794267c60cf4b40`;
- Apache-2.0;
- supports image/multi-image/video through Transformers;
- 0.5B, FP16/4-bit examples;
- English/Chinese listed; weaker reasoning/temporal counting likely, must measure.

### Excluded from first gate

- `Qwen/Qwen2.5-VL-3B-Instruct`: Qwen Research License; newer Apache-tagged Qwen3-VL-2B available.
- cloud/commercial APIs: violate offline/reproducible critical path.
- text-only Qwen planner: already rejected by repeated T4 evidence; cannot see video.
- 7B+ VLM: higher T4 resource risk; only test if 2B candidates fail quality and quantized profile is justified.

### T4 smoke gate per model

Fresh session, one candidate at a time:

- exact checkpoint SHA verified;
- built-in model class, `trust_remote_code=False`;
- FP16 on T4; BF16 not assumed;
- deterministic `do_sample=False`;
- bounded resolution/frame count/output tokens;
- peak allocated/reserved VRAM measured; operational ceiling leaves headroom for processor/query encoder;
- public synthetic matrix covers Vietnamese/English, count, OCR, action, temporal order, evidence-slot format;
- 100% hard-output parse/provenance pass on smoke matrix;
- no raw frames/prompt/generated explanation serialized;
- startup, preprocessing, generation p50/p95 measured.

Only candidates passing smoke reach labeled dev benchmark. Model-card benchmark never substitutes AIC joint score.

## Answer generation protocol

Do not rely on free-form JSON. Prior 0.5B planner evidence showed JSON schema prompts can remain malformed.

Use one bounded line:

```text
<frame_slot>\t<short_answer>
```

Example:

```text
7\t5
```

Hard parser:

- exactly one line and one tab separator;
- decimal slot in `[0, len(sampled_frame_ids))`;
- answer non-empty, single line, bounded Unicode length;
- reject control characters/extra prose;
- selected frame comes only from sampled list;
- generated text/reasoning discarded after parse;
- raw answer preserved; normalized copy only for dedup/matching.

Answer engine interface:

```python
class AnswerEngine(Protocol):
    def answer(
        self,
        query: QaQuery,
        window: EvidenceWindow,
        frames: Sequence[FrameEvidence],
        auxiliary: AuxiliaryEvidence,
    ) -> AnswerHypothesis: ...
```

No automatic hidden prompt retry. A malformed candidate fails; pipeline continues other windows. This avoids deterministic repeat of same failure.

## Automatic evidence verification

Self-reported model confidence is not trusted.

### Hard validation

- valid manifest video/frame;
- evidence slot belongs to current window;
- answer parse/type/length valid;
- OCR/ASR snippets, if present, map to same video and overlapping time;
- no duplicate normalized `(video, frame, answer)` identity;
- no more than 100 responses.

### Model support score

For top hypotheses, score two fixed suffix labels by conditional log-likelihood:

```text
SUPPORTED
UNSUPPORTED
```

Do not free-generate verifier JSON. Compute normalized token log-likelihood for both labels from same VLM/evidence prompt, then softmax into `support_score`. This is deterministic, parser-free, and avoids model self-confidence.

### Cross-window agreement

Boost only when independent evidence views agree on normalized answer:

- coarse vs refined window;
- adjacent overlapping windows;
- OCR/ASR exact support;
- never treat repeated decoding of identical prompt as independent evidence.

## OCR strategy

VLM internal OCR is baseline. External OCR remains automatic, but gated.

Candidate: PP-OCRv5 mobile/Latin through PaddleOCR.

- PaddleOCR source: Apache-2.0;
- `latin_PP-OCRv5_mobile_rec` covers most Latin-script languages, but Vietnamese is not explicitly named in docs;
- outputs recognized text + confidence;
- expects detected/cropped text lines; video sampling/tracking must be project logic;
- dependency/runtime footprint and Vietnamese diacritics accuracy require T4/CPU experiment.

Implementation order:

1. Use VLM-only baseline.
2. Build OCR adapter behind interface.
3. Trigger on text/name/sign/score answer types.
4. Run only sampled top-window frames/crops.
5. Keep text/confidence/frame provenance in memory/private evidence artifact.
6. Promote only if OCR-specific held-out joint score increases after latency/resource cost.

No global OCR precompute until error taxonomy proves need.

## ASR strategy

Candidate: `openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d`.

- 244M parameters;
- model metadata: Apache-2.0; upstream Whisper code: MIT;
- multilingual card includes `vi`;
- built-in Transformers, no remote code;
- supports Vietnamese `language="vi"`, transcription and timestamp segments;
- long-form mode requires `return_timestamps=True`; slower than short-form.

Recommended use:

- precompute timestamped ASR segments on Kaggle as separate resumable artifact when audio-error analysis justifies it;
- query time scans only top candidate videos/windows;
- map segment `[start_s,end_s]` to frame interval using canonical FPS;
- provide overlapping transcript snippets to VLM;
- never claim transcript timestamp as exact visual frame; selected submission frame still comes from decoded evidence window;
- do not serialize transcript content in aggregate reports/source bundle.

Query-time full-video transcription is too slow/unpredictable for critical path.

## Automatic modality router

Always run visual VLM. Router may add evidence:

| Question type | VLM | OCR | ASR |
|---|---:|---:|---:|
| count/action/color/object/location | yes | no by default | no |
| visible text/name/sign/score | yes | yes | no |
| spoken phrase/name/announcement | yes | optional | yes |
| unknown | yes | no initially | no initially |

Router patterns are bilingual, bounded, tested, and never remove original prompt. Unknown type falls back to visual VLM, not user interaction.

## Joint ranking and top-100 construction

For each valid hypothesis:

```text
joint_score =
    retrieval_component
  + support_component
  + cross_window_agreement
  + optional OCR/ASR support
  - duplicate/video concentration penalty
```

Rules:

- finite normalized components only;
- initial weights fixed/configured, then tune on locked train split;
- answer score cannot rescue a very weak/wrong video candidate without evidence;
- preserve raw component scores for private error analysis;
- rank verified hypotheses first, low-support valid hypotheses later rather than fabricating fallback answers;
- expand a verified answer across bounded high-score frames in same refined window to improve interval recall;
- diversify video/window/normalized-answer identities;
- stop at 100; never pad with invented responses.

If every model call/parse fails, return zero canonical responses plus aggregate failure status. Do not ask user or synthesize an ungrounded answer. The benchmark exposes this as failure.

## Failure handling

| Failure | Behavior |
|---|---|
| query split uncertain | use raw prompt for localization + question |
| multilingual encoder unavailable | fail query with aggregate status; no silent incompatible encoder swap |
| video/frame decode failure | drop window, continue others |
| VLM load failure | no answers; circuit open for session |
| one VLM window OOM/runtime failure | clear bounded cache, drop window; repeated failure opens circuit |
| malformed answer line | reject hypothesis, continue other windows |
| verifier failure | keep hard-valid hypothesis at low score or reject per registered config |
| OCR/ASR missing | continue VLM-only |
| all hypotheses fail | empty response list + private-safe aggregate failure |

No user intervention in any branch.

## Privacy and artifact policy

Allowed aggregate telemetry:

- model/processor IDs + revisions;
- frame/window counts, not pixels;
- stage elapsed p50/p95;
- peak GPU/host memory;
- parse/provenance/support pass rates;
- response count;
- source/index/manifest/config checksums.

Forbidden in source/public aggregate report:

- raw private prompt;
- frame arrays/thumbnails;
- model reasoning/free-form generated text;
- OCR/ASR transcript content;
- ground-truth answers/intervals;
- query vectors;
- credentials/cache paths/private metadata.

Official answer responses exist only in private run/submission artifact as required by competition.

## Evaluation design

### Required stage metrics

1. Event retrieval: correct-video and interval-containing-window recall at candidate depths.
2. Localization: selected evidence frame in GT interval.
3. Answer-only proxy: semantic/exact match assuming correct localization.
4. Joint Q&A: official-formula localization + answer.
5. Output integrity: parse rate, provenance pass rate, duplicate count.
6. Resources: startup, encode/retrieve/decode/preprocess/answer/verify p50/p95, peak VRAM/RAM, model/index size.

### Required ablations

- raw prompt vs event+raw retrieval variants;
- single-scale vs coarse→refine windows;
- VLM-only;
- VLM + verifier;
- VLM + OCR for OCR subset;
- VLM + ASR for audio subset;
- each model candidate under same frame/resolution/output budget.

### Promotion gate

Promote only when:

- held-out joint Final Score improves over automatic baseline;
- R@1/R@5 guardrails pass;
- output parse and provenance pass rates meet preregistered threshold;
- Vietnamese and English subsets both reported;
- T4 memory/latency fit operational budget;
- exact model/dependency/config revisions reproduce;
- no human edits used in measured pipeline.

## Implementation sequence

### Stage A — contracts and local fixtures

- `QaQuery`, `EvidenceWindow`, `FrameEvidence`, `AnswerHypothesis`, `QaPipelineStatus`.
- deterministic parser/router/window validation.
- mock answer engine.
- canonical response assembly and ranking tests.

### Stage B — retrieval and temporal evidence

- reuse `QueryEncoderRuntime`, `retrieve_kis`, manifest/video decoder;
- multi-variant wide retrieval;
- video/time clustering;
- coarse/refined exact-frame sampler;
- no real VLM dependency yet.

### Stage C — model gate notebook

- evaluate candidates separately on public synthetic clips/fixtures;
- pin winner revision and compatible Transformers version;
- record T4 VRAM/latency/protocol parse evidence;
- reject losers explicitly.

### Stage D — real VLM adapter and verifier

- lazy warm runtime;
- bounded frame/resolution/token config;
- one-line answer protocol;
- fixed-label likelihood verifier;
- fail-closed circuit state.

### Stage E — OCR/ASR automatic enrichers

- add interfaces first;
- implement only after subset/error evidence;
- ASR precompute, OCR top-window on-demand;
- provenance and resource gates.

### Stage F — CLI, Kaggle notebook and benchmark

- one command/function accepts prompt only;
- automatic pipeline writes canonical responses;
- private-safe summary;
- locked train/held-out benchmark and promotion decision.

## Recommended first implementation boundary

Implement Stages A–B and mock end-to-end tests first. In parallel only conceptually—not concurrent file edits—create a separate Kaggle model-gate notebook for Qwen3-VL-2B, SmolVLM2-2.2B and LLaVA-OneVision-0.5B. Do not add all candidate dependencies to default package. Pin only winner after fresh T4 evidence.

## Sources

- Competition contract: `docs/competition-requirements.md`
- Existing evaluation protocol: `docs/evaluation-protocol.md`
- [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl)
- [Qwen2.5-VL official blog](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [Transformers Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_vl)
- [SmolVLM2-2.2B-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Transformers SmolVLM documentation](https://huggingface.co/docs/transformers/main/en/model_doc/smolvlm)
- [LLaVA-OneVision Qwen2 0.5B model card](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- [Whisper-small model card](https://huggingface.co/openai/whisper-small)
- [Transformers Whisper documentation](https://huggingface.co/docs/transformers/main/en/model_doc/whisper)
- [PaddleOCR text recognition documentation](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html)
- [PaddleOCR license](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

## Unresolved questions

- Official semantic answer matcher/aliases vẫn chưa được công bố.
- Request 2 labeled train/held-out queries chưa có trong repo.
- Q&A query-time latency limit chưa được BTC công bố.
- Qwen3-VL-2B T4 VRAM/latency và Vietnamese quality chưa được đo.
- Competition policy cho attached model weights/Internet chưa được xác nhận.
- Batch 2 và final submission transport chưa được công bố.
