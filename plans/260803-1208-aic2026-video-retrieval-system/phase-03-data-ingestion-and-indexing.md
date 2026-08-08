---
phase: 3
title: "Data Ingestion and Indexing"
status: in-progress
priority: P1
effort: "2-4 team-days plus feature runtime"
dependencies: [1]
---

# Phase 3: Data Ingestion and Indexing

## Overview

Biến video và artifact hỗ trợ thành index tái lập, có provenance và ánh xạ exact frame. Dataset/index lớn được xử lý và lưu trong Kaggle; local chỉ giữ code, config và artifact nhỏ. Baseline chỉ cần CLIP + metadata mapping; các modality khác giữ tách biệt để không kéo complexity vào critical path.

## Requirements

- Functional: ingest nhiều batch bằng một config và canonical manifest.
- Functional: load hoặc recompute CLIP features; normalize đúng trước index.
- Functional: build/search/save/load vector index; trả về canonical candidate IDs.
- Functional: decode exact original frame/window theo yêu cầu Q&A/TRAKE/UI.
- Non-functional: incremental, resumable, deterministic; atomic output publication.
- Non-functional: không corrupt index đang dùng nếu job bị dừng.

## Architecture

```text
manifest
  ├── provided CLIP ─┐
  └── video decode ─► encoder ─► normalized matrix ─► FAISS index

index row ─► keyframe record ─► original_frame_id ─► exact-frame decoder
```

Default local hiện tại: NumPy flat exact inner-product để khóa contract và kiểm thử mà không thêm dependency. Chỉ promote FAISS `IndexFlatIP` sau khi profile Kaggle cho thấy NumPy không đạt latency/memory budget; IVF/HNSW/PQ cần thêm bằng chứng recall. Milvus ngoài phạm vi baseline.

## Related Code Files

- Create: `src/aic_retrieval/index.py`
- Create: `src/aic_retrieval/video.py`
- Create: `scripts/build_index.py`
- Create: `scripts/verify_index.py`
- Create: `tests/test_index_roundtrip.py`
- Create: `tests/test_video_frame_mapping.py`
- Create: `config/index.example.yaml`

## Implementation Steps

1. Dùng Phase 1 manifest làm input duy nhất; reject stale/incompatible manifest version.
2. Chọn provided-feature path nếu compatibility pass; ngược lại decode keyframes/video và recompute bằng model đã pin.
3. L2-normalize embeddings; assert finite values, expected dimension và row count.
4. Lưu matrix/index metadata riêng: model ID, preprocessing, dtype, normalization, manifest hash, build timestamp/version.
5. Build `IndexFlatIP` baseline cho cosine-equivalent search; giữ stable row-to-keyframe lookup.
6. Save vào staging path; reload và verify query probes trước atomic rename thành active index.
7. Implement exact-frame decoder bằng original frame ID; expose frame window iterator không materialize toàn video.
8. Cache thumbnails/windows bounded theo disk budget; cache key gồm video identity + frame + transform version.
9. Add incremental batch build: build shard độc lập rồi merge/search federated; không rebuild Batch 1 khi thêm Batch 2 nếu model/schema không đổi.
10. Verify self-nearest-neighbor cho sampled image embeddings, save/load equality và mapping round-trip.
11. Benchmark index build time, query p50/p95, peak RAM, disk size trên representative subset rồi full set.
12. Chỉ thử approximate index khi exact baseline vượt budget; đo recall loss so với exact neighbors trước promotion.

## Success Criteria

- [x] Mỗi index row ánh xạ một-một tới canonical keyframe và original frame trên synthetic manifest.
- [x] Save/reload giữ nguyên top results trên synthetic probe set.
- [x] Sampled image self-query đạt top score; duplicate-vector ties được chấp nhận và deterministic.
- [x] Staging lỗi không thay active index hợp lệ trong local test.
- [ ] Batch mới ingest được mà không đổi public retrieval contract; chờ Batch 2/Kaggle.
- [x] Exact-frame decoder khớp mocked original-frame mapping tests.
- [ ] Resource report có build time, query latency, RAM và disk size; chờ representative/full Kaggle run.

## Verification State

- Local: Kaggle source bundle chứa build/verify CLIs, clean install pass; full suite `52` tests, compile pass, Python 3.10 syntax pass trên `25` files.
- Kaggle pending: actual feature layout/model compatibility, real-codec exact seek, multi-batch build, resource profile.

## Risk Assessment

- Approximate index giảm recall tại top cutoffs. Dùng flat exact cho đến khi resource evidence buộc thay.
- Recompute features tốn GPU/time. Chunk, checkpoint, resume; giữ provided path khi chứng minh compatible.
- Video seeking theo compressed frames có thể lệch. Decode/seek verification dùng original frame count, không coi I-frame là semantic frame.

## Rollback

Active index chỉ thay bằng atomic publish sau verification. Giữ previous index metadata/path để chuyển lại ngay.
