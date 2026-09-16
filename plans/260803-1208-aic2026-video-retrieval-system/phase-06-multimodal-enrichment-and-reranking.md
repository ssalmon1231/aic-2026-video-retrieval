---
phase: 6
title: "Multimodal Enrichment and Reranking"
status: pending
priority: P2
effort: "2-6 team-days per promoted modality"
dependencies: [2, 4]
optional: true
---

# Phase 6: Multimodal Enrichment and Reranking

## Overview

Đóng khoảng trống CLIP bằng ablation có mục tiêu. Request 1 exact-visible-text failures đã đủ rõ để triển khai private OCR route trước Objects/metadata, nhưng OCR vẫn opt-in và chưa promoted. ASR/captions/VLM chỉ sau failure analysis và T4 gate. Baseline vẫn là một Python process; không port microservices hoặc Milvus.

## Requirements

- Functional: mỗi modality tạo candidate score/evidence trên canonical keyframe/video IDs.
- Functional: late-fuse với CLIP hoặc rerank shortlist; không thay frame mapping.
- Functional: lưu per-modality contribution để debug.
- Non-functional: component tắt được bằng config; baseline vẫn chạy khi artifact thiếu.
- Non-functional: promotion bắt buộc qua quality/resource gate trong `plan.md`.

## Architecture

```text
CLIP candidates ───────────────────────────┐
objects / metadata / OCR / ASR candidates ├─► normalized score fusion ─► rerank
optional grounding over shortlist ────────┘
```

Late fusion là mặc định vì feature precompute và query nhanh. Early fusion/grounding chỉ chạy trên shortlist khi late fusion không giải quyết lỗi localization.

## Related Code Files

- Create: `src/aic_retrieval/modalities.py`
- Create: `src/aic_retrieval/fusion.py`
- Create: `scripts/build_optional_features.py`
- Create: `scripts/run_ablation.py`
- Create: `tests/test_fusion.py`
- Create: `config/enrichment.example.yaml`
- Create: `plans/260803-1208-aic2026-video-retrieval-system/reports/multimodal-ablation-<modality>.md` trong lúc triển khai

## Implementation Steps

1. Dùng Phase 4 misses để đếm taxonomy theo task/cutoff; chọn đúng một modality có expected impact lớn nhất.
2. Đăng ký experiment trước: queries, split, fusion candidates/weights, quality guardrail, latency/memory ceiling.
3. Implement Objects adapter: label aliases Việt/Anh, confidence threshold, canonical keyframe mapping; giữ CLIP score khi JSON thiếu.
4. Implement nullable metadata adapter: title/description/tags nếu tồn tại; tránh over-weight YouTube text không phản ánh frame.
5. Chỉ khi OCR misses đáng kể, trích/index OCR text kèm confidence, language và frame provenance; dedup text lặp theo video.
6. Chỉ khi audio misses đáng kể, trích/index ASR segments kèm start/end timestamp; map evidence sang frame windows, không giả vờ ASR là frame evidence chính xác.
7. Chuẩn hóa score theo modality trên dev distribution; bắt đầu bằng weighted sum nhỏ, không học ranker khi dữ liệu nhãn ít.
8. Tune weights bằng search giới hạn trên train split; evaluate held-out; report delta cho cả năm cutoff và từng task.
9. Nếu late fusion còn lỗi spatial grounding, thử model early-fusion trên top-M candidates; M lấy từ latency profile.
10. Nếu audit source Vi-ATISO/AIC24 cho thấy adapter phù hợp, chỉ ADAPT parsing/API logic nhỏ với revision/license pin; không mang Docker Compose/Milvus/microservices vào baseline nếu không có scale evidence.
11. Đo index/build cost, p50/p95 latency, peak RAM/GPU, disk và offline availability.
12. Promote/reject riêng từng modality. Component reject giữ trong experiment record, không đưa vào default config.
13. Sau mỗi promotion, rerun regression cho KIS, Q&A, TRAKE; không cộng dồn model chưa có marginal-gain experiment.

## Success Criteria

- [ ] Mỗi experiment thay đổi một modality hoặc một fusion decision có thể quy trách nhiệm.
- [x] Missing private OCR artifact giữ semantic retrieval; corrupt/mismatched artifact fail closed.
- [ ] Final ranking hiển thị score contribution/provenance theo modality.
- [ ] Promoted modality tăng held-out Final Score và qua R@1/R@5/resource guardrails.
- [ ] Default config chỉ bật components đã promote.
- [ ] Early-fusion, LVLM hoặc API không chạy toàn dataset/query nếu chưa được phép và benchmark.
- [ ] Mỗi modality có quyết định promote/reject bằng số liệu.

## Risk Assessment

- Fusion weights overfit dev set. Giới hạn search, khóa split, paired query analysis.
- OCR/ASR index tăng disk/latency. Chỉ build sau evidence; dùng sparse/text index đơn giản trước.
- Metadata leakage/noise đẩy sai video lên. Cap contribution; ablate theo query type.
- Grounding model quá chậm. Shortlist-only; timeout trả lại late-fusion ranking.

## Current implementation state

- `src/aic_retrieval/ocr.py` defines private artifact descriptor/records, normalization, exact inverted lookup and bounded fuzzy search.
- `scripts/build_ocr_artifact.py` builds canonical records from keyframes; exact PaddleOCR package/model revision remains unpinned until T4 compatibility smoke.
- Hybrid RRF can recover exact OCR rows absent from semantic lists; fuzzy OCR remains bounded to semantic union.
- No held-out B3 gain measured. Default hybrid config remains disabled; OCR artifact must not be committed.

## Rollback

Tắt modality trong config để quay về frozen Phase 4 baseline. Optional indexes độc lập; không cần rebuild CLIP index.
