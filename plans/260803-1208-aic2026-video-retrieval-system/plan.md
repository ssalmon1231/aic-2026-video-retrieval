---
title: "AIC 2026 Preliminary Video Retrieval System"
description: "Kế hoạch baseline-first cho Textual KIS, Q&A và TRAKE vòng sơ tuyển AIC 2026"
status: in-progress
priority: P1
branch: "master"
tags: [aic-2026, video-retrieval, clip, qa, trake]
blockedBy: []
blocks: []
created: "2026-08-03T05:24:00.587Z"
createdBy: "ck:plan"
source: skill
---

# AIC 2026 Preliminary Video Retrieval System

## Overview

Xây dựng hệ thống code-local, data-on-Kaggle, không phụ thuộc API bên ngoài phục vụ ba loại truy vấn vòng sơ tuyển AIC 2026: Textual KIS, Q&A và TRAKE. Runtime dữ liệu/model chạy cùng Kaggle; local chỉ chạy fixtures/evaluator và artifact nhỏ. Request 2 là automatic end-to-end: người dùng chỉ nhập raw prompt, hệ thống tự retrieve, localize, answer, verify, rank và xuất canonical triples; không có human-in-the-loop trong critical path. Chiến lược: evaluator đúng trước, CLIP retrieval đủ dùng tiếp theo, automatic Q&A sau đó; chỉ promote OCR, ASR, object fusion, VLM hoặc early-fusion khi benchmark chứng minh tăng điểm trong budget.

Plan này chỉ mô tả triển khai. Trọng tâm hiện tại là đóng Request 1 — Textual KIS accuracy gate trước khi tiếp tục Request 2. Phase 7A Q&A contracts đã có; Phase 7B–7C tạm dừng. Request 1 local implementation hiện gồm raw Vietnamese B0, English multi-list RRF B1, temporal B2 và private OCR B3; promotion chỉ sau locked held-out Kaggle T4 benchmark. Request 3 — TRAKE vẫn triển khai sau khi KIS/Q&A đạt baseline dùng được. Dataset lớn nằm trên Kaggle; không tải hoặc lưu toàn bộ trên máy local. Data audit chạy trong Kaggle; local chỉ giữ code, fixtures và manifest/report nhỏ. Chưa biết deadline hoặc giao thức nộp vì tài liệu nguồn chưa công bố.

## Nguồn và mức tin cậy

### Yêu cầu chính thức

- `../../Thong tin vong So tuyen AIC2026.pdf`
  - Định nghĩa Textual KIS, Q&A, TRAKE.
  - Định dạng câu trả lời, tối đa 100 kết quả.
  - R-Score, R@1/R@5/R@20/R@50/R@100, Final Score.
  - Cấu trúc dữ liệu Batch 1; video là dữ liệu chính thức.

### Khuyến nghị tập huấn

- `../../Tập huấn AIC 2026 - Buổi 1.pptx.pdf`: pipeline indexing/retrieval, CLIP, Objects, metadata, vector database, query expansion.
- `../../Tập huấn AIC 2026 - Buổi 2.pdf`: early/late fusion, ranking, UI, deduplication, exploration/exploitation, prompt engineering.
- `../../Tập huấn AIC 2026 - Buổi 3.pdf`: agentic reasoning, tool-augmented VideoQA, RAG và temporal/spatial tools.
- Một số slide chủ yếu là hình và chưa thể kiểm chứng trực quan do môi trường thiếu trình render PDF. Không dùng các slide đó làm căn cứ cho hợp đồng bắt buộc.

## Hợp đồng cuộc thi phải giữ

| Tác vụ | Một kết quả | Điều kiện chấm |
|---|---|---|
| Textual KIS | `<video_id>, <frame_id>` | Đúng video; frame nằm trong `[s,e]` |
| Q&A | `<video_id>, <frame_id>, <answer>` | Đúng video; frame trong `[s,e]`; answer khớp ngữ nghĩa |
| TRAKE | `<video_id>, <frame_id_1>, ..., <frame_id_N>` | Sai video: 0; đúng video: tỷ lệ event frames nằm trong từng `[s_j,e_j]` |

Với mỗi `k ∈ {1,5,20,50,100}`:

```text
R@k = max(R-Score(r_i)) với 1 ≤ i ≤ k
Final Score = mean(R@1, R@5, R@20, R@50, R@100)
```

Hệ quả: candidate recall chỉ là điều kiện cần. Ranking đầu danh sách, frame mapping, deduplication và temporal refinement trực tiếp quyết định điểm.

## Phạm vi

### Bắt buộc

- Dataset manifest; kiểm tra video, keyframe, object JSON, CLIP `.npy`, metadata và mapping về original frame index.
- Evaluator tái tạo công thức chính thức; local proxy được ghi rõ cho semantic answer matching chưa được BTC công bố.
- Text-to-keyframe retrieval bằng CLIP tương thích feature được cung cấp; fallback tái trích feature từ video chính thức.
- Top-100 ranking hợp lệ; grouping/dedup không làm mất đa dạng video hoặc thời gian.
- UI/debug tooling không nằm trong critical path; production runtime nhận prompt và tự tạo ranked responses.
- Q&A tự động: query routing + retrieval + temporal evidence + VLM answer + verifier + canonical export, không user chọn frame/answer/rank.
- TRAKE video retrieval + ordered event alignment + frame-level refinement.
- Benchmark quality, latency, memory, index size; runbook và submission rehearsal.

### Chỉ triển khai qua promotion gate

- Object/metadata fusion trước; OCR/ASR khi error analysis chỉ ra khoảng trống.
- Video VLM là answer engine bắt buộc cho automatic Q&A nhưng checkpoint cụ thể chỉ được chọn qua fresh Kaggle T4 model gate; OCR/ASR/verifier là optional ablations.
- Early-fusion/grounding chỉ rerank shortlist, không chạy toàn kho.
- Agent orchestration, conversational clarification, Milvus hoặc cloud APIs không thuộc baseline.

### Ngoài phạm vi ban đầu

- Training foundation model từ đầu.
- Hạ tầng phân tán, multi-tenant, public SaaS, mobile app.
- Tự động gửi đáp án tới endpoint chưa được BTC công bố.
- Phụ thuộc bắt buộc vào Internet hoặc commercial API.

## Prior art và quyết định tái sử dụng

Repo/paper cũ là bằng chứng thiết kế, không phải nền tảng để fork nguyên trạng. Chỉ `ADAPT` code sau khi kiểm tra license ở revision dùng, dependency, contract và benchmark trên dev split AIC.

| Nguồn | Quyết định | Áp dụng |
|---|---|---|
| [TOMS Retrieval](https://github.com/ziap/toms-retrieval) — MIT | ADOPT thuật toán; REWRITE code | CLIP retrieval, ordered-query DP; thay `O(N²)` mask bằng suffix/running maximum `O(N)`, thêm backpointer/chunking/dedup |
| [Integrated Semantic and Temporal Alignment](https://arxiv.org/abs/2512.13169) | ADOPT DANTE; VERIFY claims | Same-video monotonic DP, gap penalty, backtracking, unified keyframe IDs; benchmark lại vì paper thiếu metric/ablation/resource detail |
| [Vi-ATISO](https://github.com/nxquang-al/vi-atiso) — MIT | ADOPT contracts; ADAPT chọn lọc | OCR/object/text-video modality adapters; không microservice hóa baseline |
| [AIC24](https://github.com/trnKhanh/AIC24) — MIT | AUDIT rồi ADAPT | Packaging/CLI/OCR/Milvus ideas; chưa có bằng chứng thay FAISS baseline hoặc hỗ trợ Q&A/TRAKE |
| [PIKA Search description](https://github.com/BaryuH/Ho-Chi-Minh-AI-Challenge-2025) — MIT | ADOPT architecture/UI | Hybrid retrieval và playback workflow; backend chưa công bố nên không có code để reuse |
| [BetterDay Tool](https://github.com/Nhathuy1305/BetterDay-Tool) | ADOPT concept; không copy code | Relevance feedback, shot/video browsing; license lineage mâu thuẫn và native stack không phù hợp baseline Python |
| [vitrivr-ng](https://github.com/vitrivr/vitrivr-ng) — MIT | ADOPT UI patterns | Query-by-example, browse/timeline interaction; frontend phụ thuộc Cineast, không dùng làm core |
| [SOMHunter](https://github.com/siret/somhunter) — GPL-2.0 | Concept-only | SOM/relevance-feedback experiment; không copy code vào project nếu không chấp nhận GPL |
| [Moment-DETR](https://github.com/jayleicn/moment_detr) — MIT | Optional experiment | Temporal interval proposal/saliency; clip khoảng 2 giây không đủ exact-frame TRAKE |
| [UniVTG](https://github.com/showlab/UniVTG) — MIT | Optional experiment | Temporal grounding reranker; cần feature/checkpoint khác và benchmark Kaggle trước promotion |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) — Apache-2.0 metadata; [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) — Apache-2.0; [LLaVA-OneVision 0.5B](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) — Apache-2.0 | T4 model-gate candidates | Automatic answer engine đọc bounded exact-frame windows; promote đúng một checkpoint qua joint Q&A/reliability/resource gate |
| [Whisper-small](https://huggingface.co/openai/whisper-small) — Apache-2.0 metadata; [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Apache-2.0 | Optional automatic evidence | Timestamped Vietnamese ASR và OCR trên top evidence windows; không thay visual exact-frame provenance |
| [LightVideoRAG](https://github.com/linshys/lightvideoRAG) — Apache-2.0; [LongVideoBench](https://github.com/longvideobench/LongVideoBench) | Audit/concept only | Adaptive sampling và metadata-aware evaluation ideas; chưa dùng làm dependency hoặc bằng chứng exact-frame AIC |
| [VideoRAG](https://github.com/starsuzi/VideoRAG); [AgenticVideoRAG](https://github.com/Lam810/AgenticVideoRAG-public) | Idea/watchlist only | Corpus/iterative evidence retrieval; thiếu license hoặc exact-frame evidence/benchmark đủ mạnh, không copy code |

Thứ tự áp dụng: shared contracts/evaluator → CLIP KIS → automatic Q&A contracts/evidence → T4 VLM model gate → verifier/modality ablations → TRAKE implementation. Không thêm Milvus, microservices, SOM hoặc agent nếu fixed automatic pipeline vẫn đạt budget.

## Kiến trúc mục tiêu tối thiểu

```text
Official videos + support artifacts
        │
        ▼
Manifest / validation / frame mapping
        │
        ├── CLIP matrix ──► exact index
        ├── optional OCR/ASR artifacts
        └── exact-frame decoder
                 │
raw prompt ─► deterministic routing ─► retrieve ─► temporal evidence windows
                                                    │
                                                    ├── automatic video VLM answer
                                                    ├── automatic evidence verifier
                                                    └── TRAKE alignment
                                                             │
                                                    canonical ranked responses
                                                             │
                                                   evaluator/submission adapter
```

Mặc định đề xuất: Python, NumPy exact index, pinned multilingual text encoder và đúng một video VLM được promote trên NVIDIA T4. UI chỉ phục vụ debug/inspection, không tạo hoặc sửa production responses. Phase 1/3/4 phải xác minh feature compatibility, frame mapping, package compatibility và hardware trước khi khóa dependency.

## Phases

Thứ tự thực hiện hiện tại: `Request 1 B0–B3 integration/review → locked Kaggle T4 benchmark/promotion → 7B → 7C model gate → 7D–7H`; Phase 5 UI không block automatic Q&A, Phase 6 OCR slice đã được triển khai opt-in nhưng chưa promoted, Phase 8 sau baseline Request 1–2, Phase 9 sau khi cả ba request sẵn sàng.

| Phase | Name | Priority | Status |
|-------|------|----------|--------|
| 1 | [Requirements and Data Audit](./phase-01-requirements-and-data-audit.md) | P1 | In progress — Kaggle source bundle ready; official data audit pending |
| 2 | [Evaluation and Experiment Protocol](./phase-02-evaluation-and-experiment-protocol.md) | P1 | Completed |
| 3 | [Data Ingestion and Indexing](./phase-03-data-ingestion-and-indexing.md) | P1 | In progress — build/verify included in Kaggle bundle; real build/profile pending |
| 4 | [CLIP Retrieval Baseline](./phase-04-clip-retrieval-baseline.md) | P1 | In progress — B0–B2 local implementation/integration present; locked Kaggle metrics pending |
| 5 | [Retrieval Interface and Submission Workflow](./phase-05-retrieval-interface-and-submission-workflow.md) | P1 | Pending — debug/inspection UI optional for Request 2 runtime |
| 7 | [Automatic Q&A Pipeline](./phase-07-q-a-pipeline.md) | P1 | Paused after 7A — wait for Request 1 promotion gate |
| 6 | [Multimodal Enrichment and Reranking](./phase-06-multimodal-enrichment-and-reranking.md) | P2/optional | In progress — private OCR B3 local implementation present; T4/held-out gate pending |
| 8 | [TRAKE Temporal Alignment](./phase-08-trake-temporal-alignment.md) | P2 | Pending |
| 9 | [System Optimization and Competition Readiness](./phase-09-system-optimization-and-competition-readiness.md) | P1 after 7/8 | Pending |

## Dependencies

```text
1 ─┬─► 2 ─┬─► 4 ─┬─► 5 (KIS/debug UI, không block automatic Q&A)
   └─► 3 ─┘      ├─► 7 automatic Q&A ─┐
                  ├─► 6 optional ──────┼─► 9
                  └─► 8 TRAKE ─────────┘
```

- Phase 2 và 3 có thể chạy song song sau Phase 1.
- Phase 7 contracts/evidence phụ thuộc Phase 2/3/4, không phụ thuộc human UI Phase 5.
- Phase 5 và optional Phase 6 có thể chạy song song; Phase 5 không được thêm answer/frame editing vào Request 2 production path.
- Phase 7 dùng đúng một VLM được chọn qua T4 model gate; OCR/ASR chỉ dùng sau promotion riêng.
- Phase 8 triển khai sau khi Request 1–2 có baseline dùng được; thiết kế/fixtures có thể chuẩn bị trước.
- Không có plan khác trong repo tạo dependency chéo.

## Promotion Gate

Mỗi nâng cấp phải có một experiment record cố định: baseline commit/config, dataset split, seed nếu có, metric theo từng tác vụ/cutoff, p50/p95 latency, peak memory, index size và failure cases.

Chỉ promote khi:

1. Final Score proxy tăng trên dev set đã khóa.
2. R@1 và R@5 không giảm vượt ngưỡng được đăng ký trước experiment.
3. Latency/memory đáp ứng budget được chốt sau audit hardware.
4. Kết quả tái lập được từ manifest + config.
5. Độ phức tạp vận hành phù hợp thời gian thi.

## Acceptance Criteria Toàn Hệ Thống

- [ ] Ingest Batch 1 và Batch 2 bằng cùng pipeline; artifact thiếu tạo cảnh báo, không làm hỏng toàn bộ run.
- [ ] Mọi candidate truy ngược được tới video, keyframe và exact original frame.
- [ ] Evaluator qua unit tests cho ví dụ chính thức: KIS 0/1, Q&A 0/1, TRAKE `3/4 = 0.75`, Final Score `0.74`.
- [ ] Ba workflow tạo tối đa 100 ranked responses đúng schema.
- [ ] Request 2 nhận raw prompt duy nhất và tự động retrieve, localize, answer, verify, rank, export; không có human edits trong measured/production path.
- [ ] Benchmark report có quality/latency/memory/index size theo task.
- [ ] Full offline rehearsal chạy từ dữ liệu mới đến file nộp; validator không báo lỗi.
- [ ] Mỗi component ngoài baseline có quyết định promote/reject dựa trên số liệu.

## Rủi ro chính

| Rủi ro | Giảm thiểu |
|---|---|
| Provided CLIP feature không tương thích text encoder | Xác minh dimension/model/preprocessing; test similarity; fallback recompute từ video |
| Keyframe mapping sai original frame | Kiểm tra schema; decode sample frame; round-trip test; UI luôn hiển thị original frame ID |
| Semantic answer evaluator của BTC không công khai | Tách exact official conditions khỏi local semantic proxy; giữ raw answer; adapter thay thế được |
| TRAKE interval thường dưới 10 frame | Refine trên every-frame window; monotonic alignment; không dùng I-frame như semantic frame |
| Batch 2 hoặc artifact hỗ trợ thay đổi | Manifest versioned; tolerant parser; video là source of truth |
| Hardware/latency chưa biết | Profile sớm; local CPU fallback; khóa budget khi BTC/team xác nhận |
| Model/API policy chưa biết | Automatic Q&A dùng attached/cached local VLM, không cloud API; freeze offline model artifact sau T4 gate |
| VLM hallucination/malformed output | Project-owned frame slots, one-line protocol, hard provenance, fixed-label verifier, empty-output fail closed |
| T4 memory/latency chưa đủ | 2B-first candidate gate, bounded frames/resolution/tokens, one model/session, reject khi vượt budget |
| Automatic rank trả near-duplicate frame | Temporal clustering + video/window/answer diversity; giữ raw candidate pool để tránh mất recall |

## Request 2 automatic Q&A decision

- Người dùng chỉ nhập raw prompt; không chọn frame, nhập/sửa answer hoặc chỉnh rank.
- Automatic pipeline là retrieval → temporal evidence → video VLM → verifier → canonical triples.
- T4 model gate candidates và implementation chi tiết nằm tại `phase-07-q-a-pipeline.md`.
- Research evidence: `../reports/research-260808-2105-automatic-qa-request-2-report.md`.
- First implementation boundary: Phase 7A–7B contracts/query routing/retrieval/evidence với mock model; real checkpoint chỉ pin sau fresh T4 gate.

## Câu hỏi chưa được giải quyết

1. Deadline và lịch công bố Batch 2?
2. Submission thực tế qua API, CSV hay giao diện; schema/encoding cụ thể?
3. Latency limit, số truy vấn và số lượt nộp?
4. Hardware ngày thi có giữ NVIDIA T4 16 GB như runtime mặc định hiện tại không?
5. Request 2 labeled train/held-out queries và ground truth có được cung cấp không?
6. Attached/cached model weights và offline runtime có được phép trong thời gian thi không?
7. Exact semantic-answer matcher của BTC xử lý song ngữ, số, đơn vị và alias thế nào?
8. Qwen3-VL-2B, SmolVLM2-2.2B hoặc LLaVA-OneVision-0.5B checkpoint nào qua T4 joint-Q&A/resource gate?
