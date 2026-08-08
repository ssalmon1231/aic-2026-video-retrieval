---
phase: 8
title: "TRAKE Temporal Alignment"
status: pending
priority: P2
effort: "4-7 team-days"
dependencies: [2, 3, 4, 5]
optional_dependencies: [6]
---

# Phase 8: TRAKE Temporal Alignment

## Overview

Thiết kế Request 3 — TRAKE sau baseline Request 1–2: tìm đúng video và một exact semantic frame cho từng event theo thứ tự. Minimum correct baseline dùng DANTE-style same-video monotonic DP, backpointers và coarse-to-fine original-frame refinement; learned temporal grounding chỉ qua promotion gate.

## Requirements

- Functional: parse ordered list `N` events; giữ count/order không đổi.
- Functional: rank candidate videos/sequences; trả tối đa 100 `<video_id, frame_1...frame_N>`.
- Functional: partial event matches được evaluator ghi nhận.
- Functional: refine từ sparse keyframe đến exact original frame; target intervals thường dưới 10 frames.
- Non-functional: alignment deterministic; timeout có best-so-far result.

## Architecture

```text
events ─► event embeddings ─► per-event keyframe candidates
whole sequence ─► candidate videos
candidate video timelines + event scores
          ─► DANTE-style monotonic DP + gap penalty + backpointers
          ─► top sequence hypotheses
          ─► original-frame local refinement
          ─► ranked sequence candidates
```

Với event `i`, timeline position `t`, baseline dùng recurrence tương đương `DP[i,t] = S[i,t] + max_{τ<t}(DP[i-1,τ] - λ(t-τ))`; running maximum giảm từ quadratic xuống tuyến tính theo timeline mỗi event. `λ` là hyperparameter phải tune, không copy giá trị paper. Không dùng technical I-frame làm answer. Keyframes chỉ tạo coarse candidates; original video là nguồn exact frame. Moment-DETR/UniVTG chỉ là optional interval-proposal/reranking experiments: feature mismatch và clip-level resolution không đáp ứng exact-frame contract mặc định.

## Related Code Files

- Create: `src/aic_retrieval/trake.py`
- Create: `src/aic_retrieval/alignment.py`
- Create: `scripts/align_events.py`
- Create: `tests/test_temporal_alignment.py`
- Create: `config/trake-baseline.yaml`
- Update: `src/aic_retrieval/app.py`
- Update: `src/aic_retrieval/submission.py`

## Implementation Steps

1. Định nghĩa TRAKE query record: raw text + ordered events; UI cho phép sửa segmentation nhưng không reorder ngầm.
2. Encode từng event và whole sequence; retrieve wide per-event keyframes trên toàn corpus.
3. Aggregate candidate-video score từ event coverage, best event scores và sequence query; ưu tiên video có evidence cho nhiều events.
4. Với mỗi candidate video, tạo dense/sparse timeline score `S[event, position]`; merge near-duplicate coarse candidates thành windows nhưng giữ score/profile.
5. Implement strict same-video monotonic DP bằng running maximum, gap penalty và backpointers. Invalid transitions dùng `-∞`; xác minh recurrence với brute-force enumerator trên fixtures nhỏ.
6. Mở rộng top-K paths hoặc bounded beam chỉ sau best-path baseline; global-rank giữa videos để tận dụng top-100 mà không nổ tổ hợp.
7. Refine mỗi aligned keyframe trong local original-frame window. Search every frame hoặc adaptive window theo interval/hardware; rescore event-frame similarity.
8. Re-run local monotonic consistency sau refinement; không chọn independent argmax làm đảo thứ tự.
9. Chọn one semantic frame per event; map tới original frame IDs; validate event count, video identity, frame bounds và strict order theo contract đã xác minh.
10. Evaluate video accuracy, event hit rate, partial R-Score distribution và Final Score theo cutoff; lưu path visualization cho failures.
11. Tune candidate video depth, top-path/beam width, gap penalty `λ` và refinement radius trên train split; không dùng trực tiếp `0.001/0.01` từ DANTE paper nếu AIC dev data không xác nhận.
12. Optional experiment: Moment-DETR hoặc UniVTG đề xuất/rerank temporal windows trên shortlist video; đo feature-recompute cost, interval recall, exact-frame gain và latency trước promotion.
13. UI hiển thị event rows + linked timeline; cho chỉnh exact frames nhưng cảnh báo order/frame-bound violations trước export.

## Success Criteria

- [ ] Sequence response luôn có đúng `N` frames trên cùng một video.
- [ ] Unit tests cover wrong video zero, all/partial/no events, narrow intervals, duplicate candidates và order violations.
- [ ] Linear DP score/path khớp brute-force trên random fixtures nhỏ; peak memory không chứa timeline-by-timeline quadratic matrix.
- [ ] Fixture `3/4` matched events trả R-Score `0.75`.
- [ ] Refined frames là original frame IDs, không phải keyframe ordinal hoặc compression I-frame.
- [ ] Top-100 chứa nhiều sequence/video hypotheses hợp lệ, có deterministic order.
- [ ] Report có video accuracy, per-event accuracy, partial score, Final Score và p95 latency.
- [ ] Timeout trả best verified hypotheses, không trả malformed partial sequence.

## Risk Assessment

- Mô tả event có thể không theo chronological order. Phase 1 phải xác minh contract; nếu mơ hồ, expose ordered/unordered mode nhưng mặc định theo đề.
- Sparse keyframes bỏ lỡ interval <10 frames. Dùng video-level evidence + local every-frame refinement.
- Beam width nổ tổ hợp. Bound per-event candidates/video/beam; profile; giữ best-so-far.
- Independent event retrieval chọn nhiều video. Candidate-video coverage score trước alignment; response bắt buộc một video.

## Rollback

Tắt fine refinement hoặc giảm beam để quay về coarse monotonic baseline khi latency quá cao. Canonical sequence schema không đổi.
