---
phase: 5
title: "Retrieval Interface and Submission Workflow"
status: pending
priority: P1
effort: "2-4 team-days"
dependencies: [3, 4]
---

# Phase 5: Retrieval Interface and Submission Workflow

## Overview

Tạo UI/debug process chạy cạnh dataset/index trên Kaggle để inspect retrieval, timeline và exact-frame provenance; cùng app chạy local với fixture/artifact nhỏ để phát triển. Production Request 2 không phụ thuộc UI: raw prompt đi qua automatic Q&A pipeline và output không được sửa bằng human interaction. Submission adapter tách khỏi UI vì BTC chưa công bố protocol cuối cùng.

## Requirements

- Functional: nhập query; xem ranked grid, video groups, timeline và playback quanh candidate.
- Functional: inspect canonical responses cho cả ba task; Q&A production responses là read-only và được tạo tự động bởi Phase 7.
- Functional: relevance feedback tùy chọn rerank candidate pool hiện có; không rebuild index hoặc đổi canonical score provenance.
- Functional: validate, preview, lưu draft và export tối đa 100 ranked responses.
- Non-functional: hoạt động không cần API/model bên ngoài; không mất lựa chọn khi refresh hoặc retrieval lỗi.
- Non-functional: thao tác bàn phím, focus rõ, text labels và contrast đủ dùng.

## Architecture

```text
local UI ─► retrieval service ─► canonical candidates
   │              │
   ├── timeline/exact-frame decoder
   ├── task-specific response editor
   └── draft store ─► submission adapter ─► validator ─► export
```

Chọn framework có sẵn hoặc nhỏ nhất đáp ứng playback/state. Không thêm frontend/backend tách riêng nếu một local process đủ dùng.

## Related Code Files

- Create: `src/aic_retrieval/app.py`
- Create: `src/aic_retrieval/ui_state.py`
- Update: `src/aic_retrieval/submission.py`
- Create: `tests/test_submission_workflow.py`
- Create: `config/ui.example.yaml`
- Create: `README.md` với lệnh local run sau khi implementation tồn tại

## Implementation Steps

1. Xác định interaction budget từ số query/thời gian thi; wireframe một màn hình chính thay vì nhiều route.
2. Implement query form với task selector; giữ raw query và lịch sử phiên local.
3. Render result grid gồm rank, score, video ID, original frame ID, thumbnail; group/collapse theo video nhưng giữ global rank.
4. Thêm timeline quanh candidate: neighboring keyframes, exact-frame stepping, seek/playback theo timestamp; luôn hiển thị original frame ID dùng để nộp.
5. Thêm keyboard navigation, select/unselect, move rank, open video, previous/next frame; tránh thao tác chuột lặp lại.
6. Tạo task views: KIS/TRAKE editor chỉ nếu competition workflow cần; Q&A view read-only hiển thị automatic answer, exact evidence frame và score/provenance, không cho sửa measured/production output.
7. Sau frozen baseline, thử relevance feedback tối thiểu từ BetterDay/SOMHunter concept: positive/negative selected frames rerank raw pool bằng embedding centroid/score adjustment. Feature tắt mặc định; không copy GPL/unclear-license code; promote chỉ khi interaction rehearsal tăng điểm hoặc giảm thời gian.
8. Persist draft atomically theo query/task; restore sau restart; không overwrite draft khác khi query ID trùng mà nội dung khác.
9. Dùng một canonical response model từ Phase 2; UI không tự tạo schema riêng.
10. Validate required fields, frame/video existence, event count/order, duplicate policy, rank và giới hạn 100 trước export.
11. Tạo submission adapter có interface cố định, formatter cụ thể bổ sung khi BTC công bố API/CSV/UI protocol.
12. Export cùng manifest: query ID, task, generated time, adapter version, index/config identity; không tự gửi ra ngoài.
13. Chạy usability rehearsal cho Request 1/TRAKE UI; với Request 2 chỉ đo automatic prompt-to-output latency/failures và read-only inspection, không ghi thao tác chỉnh output.

## Success Criteria

- [ ] Một command mở được UI cạnh Kaggle-mounted dataset/index; local fixture mode không cần Internet.
- [ ] Candidate từ UI truy ngược đúng video, keyframe và original frame.
- [ ] Draft phục hồi được sau restart và không bị mất khi retrieval lỗi.
- [ ] KIS/TRAKE debug/editor flow và Q&A read-only inspection cùng dùng canonical responses; Q&A output được evaluator/validator chấp nhận mà không human edits.
- [ ] Export >100, missing field, wrong event count hoặc invalid frame bị chặn với lỗi cụ thể.
- [ ] Keyboard-only flow hoàn thành được query, review, selection và export.
- [ ] Relevance-feedback experiment tắt được; nếu promote phải có paired quality/time evidence.
- [ ] Rehearsal report có task time và failure points.

## Risk Assessment

- Framework nặng làm chậm startup/đóng gói. Ưu tiên dependency đã có; một process local.
- Grouping che mất global ranking. Hiển thị cả group và rank; cho chuyển sang ungrouped view.
- Frame stepping sai do FPS/seek. Dùng exact-frame decoder đã kiểm thử ở Phase 3.
- Protocol chưa biết. Không encode assumptions vào UI; thay formatter qua adapter.

## Rollback

UI chỉ quản lý drafts và adapter. CLI retrieval/export vẫn là fallback; giữ draft schema versioned để migrate hoặc export thủ công.
