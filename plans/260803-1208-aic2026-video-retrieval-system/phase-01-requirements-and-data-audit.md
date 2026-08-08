---
phase: 1
title: "Requirements and Data Audit"
status: in-progress
priority: P1
effort: "1-2 team-days after data access"
dependencies: []
---

# Phase 1: Requirements and Data Audit

## Overview

Chuyển tài liệu thành contracts kiểm thử được; kiểm kê dữ liệu thật trước khi chọn dependency hoặc kiến trúc chi tiết. Kết quả phase này khóa ID conventions, frame mapping, artifact compatibility và các unknown cần BTC xác nhận.

## Requirements

- Functional: mô tả chính xác input/output của Textual KIS, Q&A, TRAKE; tối đa 100 ranked responses.
- Functional: kiểm kê Videos, Keyframes, Objects, CLIP features, Metadata theo batch/video.
- Functional: xác minh ánh xạ keyframe index sang original frame index.
- Non-functional: video là source of truth; missing support artifacts chỉ cảnh báo.
- Non-functional: dependency/code reuse phải có source revision, license và quyết định ADOPT/ADAPT/REJECT.
- Non-functional: không ghi dữ liệu cá nhân hoặc sample frames vào log/report công khai.

## Architecture

Tạo canonical manifest, một record cho mỗi video/keyframe. Mọi downstream module chỉ dùng canonical IDs, không tự suy luận từ đường dẫn. Validator phân biệt `error` làm kết quả không thể tin cậy với `warning` cho artifact hỗ trợ thiếu.

Canonical fields tối thiểu:

```text
video_id, video_path, fps, frame_count, duration
keyframe_id, keyframe_path, keyframe_ordinal, original_frame_id
clip_row, object_path, metadata_path, batch_id
```

## Related Code Files

- Create: `docs/competition-requirements.md`
- Create: `config/dataset.example.yaml`
- Create: `src/aic_retrieval/contracts.py`
- Create: `src/aic_retrieval/data.py`
- Create: `scripts/audit_dataset.py`
- Create: `tests/test_data_contracts.py`
- Preserve: bốn PDF nguồn ở project root

## Implementation Steps

1. Ghi requirements matrix: task, query fields, response fields, score conditions, top-k cutoffs, maximum result count.
2. Ghi riêng facts chính thức, khuyến nghị tập huấn, assumptions nội bộ; cấm nâng assumptions thành luật BTC.
3. Giữ dataset trên Kaggle; chạy audit script/notebook ngay trong Kaggle environment. Không tải hoặc lưu toàn bộ video/features/keyframes trên máy local.
4. Inventory từng batch từ Kaggle-mounted paths: số video, keyframe folders, object JSON, feature arrays, metadata files; phát hiện orphan/duplicate/missing records.
5. Probe video bằng decoder: FPS, frame count, duration, codec; ghi lỗi đọc file.
6. Probe CLIP `.npy`: shape, dtype, normalization, row count; đối chiếu row với keyframe order.
7. Probe Objects: schema, category/score/bbox ranges, tối đa 100 detections; không giả định file luôn tồn tại.
8. Probe metadata: encoding/schema tùy biến; các field YouTube đều nullable.
9. Xác định mapping source cho `original_frame_id`; round-trip ít nhất một mẫu đầu/giữa/cuối mỗi batch bằng decode exact frame và so với keyframe.
10. Xác minh text encoder tương thích provided `clip-ViT-B-32`: dimension, normalization, tokenizer/model identity; nếu không chứng minh được, đánh dấu recompute-required.
11. Xuất audit summary machine-readable; report chỉ chứa aggregate/error paths đã làm sạch.
12. Chốt prior-art register từ `plan.md`: source URL/revision, license, task mapping, ADOPT/ADAPT/REJECT, dependency/resource cost và experiment gate. Không copy code có GPL, unknown hoặc conflicting license.
13. Chốt dependency/runtime matrix sau khi biết Kaggle image, Python, GPU/CUDA, RAM, disk; giữ CPU fallback và pin revision cho mọi code được ADAPT.

## Success Criteria

- [x] Requirements matrix bao phủ cả ba task và đúng công thức tài liệu chính thức.
- [ ] 100% video có canonical manifest record; support artifacts thiếu được liệt kê. Cần chạy dataset thật trên Kaggle.
- [ ] Mọi CLIP row ánh xạ một-một tới keyframe hoặc dataset bị fail-fast với lỗi cụ thể. Logic đã test; cần audit dataset thật.
- [ ] Frame mapping vượt round-trip checks trên mẫu đầu/giữa/cuối. Logic đã test; cần video thật trên Kaggle.
- [x] Metadata thiếu không chặn ingestion.
- [ ] Có quyết định dùng provided features hay recompute, kèm bằng chứng compatibility. Cần model/preprocessing identity và empirical check trên Kaggle.
- [x] Mọi prior-art dependency/code reuse có revision, license và decision record; nguồn chưa rõ license chỉ dùng concept. Chưa ADAPT source code ngoài.
- [x] Dataset và private frame content nằm ngoài git.
- [x] Deterministic source-only Kaggle bundle và runbook đã verified; bundle không chứa PDF/data/vector/index/label/private artifacts.

## Verification State

- Local: source bundle build, deterministic checksum, archive allowlist, clean install và audit CLI help đã pass.
- Kaggle pending: official dataset mount/path/schema, full inventory, sampled frame mapping và CLIP provenance.

## Risk Assessment

- Mapping có thể dùng filename ordinal thay vì original frame ID. Không cho downstream dùng ordinal trước khi round-trip pass.
- Batch 2 có thể đổi schema. Parser tolerant với optional fields; strict với identity/mapping fields.
- Feature model name chưa đủ xác định implementation. So sánh bằng empirical embedding test, không dựa vào tên.

## Rollback

Không mutate dữ liệu nguồn. Xóa generated manifest/cache rồi chạy audit lại bằng config sửa đổi.
