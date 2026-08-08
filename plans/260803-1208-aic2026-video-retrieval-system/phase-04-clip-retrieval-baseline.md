---
phase: 4
title: "CLIP Retrieval Baseline"
status: in-progress
priority: P1
effort: "2-3 team-days"
dependencies: [2, 3]
---

# Phase 4: CLIP Retrieval Baseline

## Overview

Tạo baseline Request 1 — Textual KIS end-to-end: query text đến top-100 valid frame responses. Dùng ý tưởng CLIP retrieval từ TOMS nhưng rewrite tối thiểu quanh canonical manifest/FAISS; tối ưu ranking và temporal/video diversity trước khi thêm model mới.

## Requirements

- Functional: encode truy vấn Việt/Anh bằng text encoder tương thích image features.
- Functional: retrieve candidate pool, rerank, temporal deduplicate/diversify, xuất tối đa 100 results.
- Functional: hỗ trợ optional ordered subqueries bằng same-video monotonic alignment tuyến tính; single-query path không chịu thêm complexity.
- Functional: giữ raw scores/candidates cho error analysis.
- Non-functional: deterministic với cùng index/config; p95 latency được đo.
- Non-functional: không gọi LLM/cloud trong baseline.

## Architecture

```text
query ─► normalize ─► prompt variants ─► text embeddings
      ─► chunked vector search (wide pool) ─► score aggregation
      ├── single query ────────────────────────────────────┐
      └── ordered subqueries ─► monotonic DP/backpointers ├─► temporal clustering/video diversity ─► ranked top 100
```

Dùng vài prompt templates cố định, audit được; không tạo agent. Candidate pool lớn hơn 100 để dedup/ranking có chỗ chọn, nhưng giá trị được profile thay vì hard-code mù. Không port TOMS nguyên trạng: tránh full-corpus GPU residency, triangular `O(N²)` mask, fixed top-1000, import-time initialization và raw HTML interpolation.

## Related Code Files

- Create: `src/aic_retrieval/query.py`
- Create: `src/aic_retrieval/retrieval.py`
- Create: `src/aic_retrieval/ranking.py`
- Create: `scripts/search.py`
- Create: `tests/test_retrieval.py`
- Create: `config/retrieval-baseline.yaml`
- Create: `src/aic_retrieval/benchmark.py`
- Create: `scripts/benchmark_kis.py`
- Create: `tests/test_kis_benchmark.py`
- Create: `config/kis-benchmark.example.yaml`

## Implementation Steps

1. Implement query normalization bảo toàn nội dung gốc; Unicode whitespace/punctuation cleanup không dịch hoặc bỏ negation.
2. So sánh query gốc, prompt template Việt/Anh và mean/max ensemble nhỏ trên dev set; giữ template thắng qua promotion gate.
3. Encode + normalize text embedding bằng cùng model family đã xác minh ở Phase 1.
4. Search wide top-N keyframes theo chunks/index; preserve similarity, index rank, video ID, keyframe ID, original frame ID. Candidate depth là config được benchmark, không fixed top-1000.
5. Aggregate multiple prompt variants bằng phương pháp đơn giản đã đăng ký trước; baseline đầu tiên dùng mean normalized score.
6. Nếu query chứa ordered subqueries rõ ràng, group scores theo video rồi dùng TOMS/DANTE-style monotonic DP với running/suffix maximum, invalid state `-∞` và backpointers; complexity tuyến tính theo timeline cho mỗi subquery, không tạo triangular `O(N²)` mask. Không tự tách query thường thành chuỗi bằng LLM.
7. Cluster near-duplicate frames trong cùng video bằng temporal distance; chọn representative tốt nhất nhưng giữ members cho timeline.
8. Diversify early ranks giữa video/temporal regions với penalty cấu hình được; không hard-filter một result mỗi video vì GT có thể nằm ở cụm khác.
9. Convert candidates sang KIS responses bằng original frame ID; validate schema và top-100 limit.
10. Run evaluator theo query, cutoff, video; lưu misses theo taxonomy: semantic gap, object/action, text/OCR, audio, temporal order, mapping, ranking.
11. Tune chỉ candidate depth, prompt set, optional gap penalty và diversity trên training split; evaluate một lần trên held-out split. Relevance feedback giữ ở UI experiment sau baseline, không đổi initial ranking.
12. Profile latency từng stage; cache query embeddings theo normalized query + model version.
13. Freeze baseline config/report trước Phase 5-8 để mọi nâng cấp có comparator.

## Success Criteria

- [x] CLI query-vector trả ranked results có valid video/original frame IDs trên synthetic index.
- [x] Validator chấp nhận danh sách 1-100 KIS responses.
- [x] Raw candidate ranking và final ranking đều truy vết được.
- [ ] Image features đã empirically match OpenAI CLIP ViT-B/32; repository raw-text path đã nối với pinned revision nhưng còn chờ Kaggle runtime gate và ranked-output comparison.
- [ ] Ordered-query DP/backpointers deferred; chỉ triển khai sau single-query benchmark và có brute-force oracle.
- [ ] Temporal dedup đúng deterministic local contract; tác động held-out Final Score chờ dev set Kaggle.
- [x] Batch benchmark local nhận ordered `.npy` query vectors + labeled KIS query set; xuất R@1/R@5/R@20/R@50/R@100, Final Score, p50/p95, peak RSS nullable, index size và provenance.
- [ ] Real baseline metrics chưa có; chờ labeled queries, encoder đã xác minh và runtime Kaggle.
- [x] Không có cloud/API/model dependency trong local vector baseline.
- [x] Baseline config, index metadata và raw/final output đủ tái chạy local.

## Verification State

- Local: full suite `76` tests pass; raw-text tests mock tokenizer/model nên không download Hugging Face; vector-only CLI compatibility và query non-serialization đều pass.
- Verification: `compileall`, Python 3.10 AST parse trên `28` files và `git diff --check` pass. Deterministic source bundle build hai lần cùng SHA-256 `3e7e52e1580de01e4538303fd062c799ccfdc25e59735f0fcc04e70387cb32d6`.
- Kaggle evidence hiện có: image features empirically match `openai/clip-vit-base-patch32`; exact index verified; English manual smoke khoảng 9/10 relevant. Repository CLI còn chờ xác nhận `_commit_hash`, ranked-output comparison, real Textual KIS labels, held-out metrics và full latency/RSS profile.

## Risk Assessment

- CLIP yếu với hành động nhỏ, OCR, audio, temporal order. Gắn failure taxonomy; xử lý có mục tiêu ở Phase 6-8.
- Dedup/diversity có thể đẩy đúng frame xuống. Tune theo metric cutoffs; lưu raw pool để phục hồi.
- Prompt expansion có thể drift ý nghĩa. Giới hạn templates, giữ query gốc, kiểm tra negation/entity.

## Rollback

Mọi ranking transform config-driven. Tắt prompt ensemble/diversity để quay về raw cosine baseline mà không rebuild index.
