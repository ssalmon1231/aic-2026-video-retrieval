---
phase: 2
title: "Evaluation and Experiment Protocol"
status: completed
priority: P1
effort: "1-2 team-days"
dependencies: [1]
---

# Phase 2: Evaluation and Experiment Protocol

## Overview

Xây evaluator trước retrieval để mọi thay đổi tối ưu đúng Final Score. Tách công thức chính thức có thể tái tạo chính xác khỏi semantic-answer proxy chưa được BTC công bố.

## Requirements

- Functional: score Textual KIS, Q&A, TRAKE; hỗ trợ tối đa 100 ranked responses.
- Functional: tính R@1, R@5, R@20, R@50, R@100 và Final Score.
- Functional: báo metric theo query type, cutoff, query và aggregate.
- Non-functional: deterministic, testable, không silently truncate/repair submission sai.
- Non-functional: experiment comparison chứa config/data/code identity và resource metrics.

## Architecture

```text
ground-truth loader ─┐
submission loader ───┼─► schema validation ─► task scorer ─► top-k reducer ─► report
answer matcher ──────┘
```

Q&A matcher là strategy rõ nhãn:

- `exact`: test công thức và normalized equality.
- `local-semantic-proxy`: alias/song ngữ/số/đơn vị do dev set định nghĩa.
- `official`: adapter trống cho matcher BTC nếu được cung cấp.

Không báo proxy score như điểm BTC chính xác.

## Related Code Files

- Create: `src/aic_retrieval/evaluation.py`
- Create: `src/aic_retrieval/submission.py`
- Create: `src/aic_retrieval/experiment.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/fixtures/official-examples.json`
- Create: `docs/evaluation-protocol.md`

## Implementation Steps

1. Định nghĩa typed records cho ground truth và response của từng task.
2. Validate required fields, data type, rank order, duplicate policy, event count và giới hạn 100.
3. Implement KIS indicator: đúng video và `frame_id ∈ [s,e]`.
4. Implement Q&A indicator: KIS conditions cộng answer matcher; lưu raw answer trước normalization.
5. Implement TRAKE: sai video bằng 0; đúng video bằng số event frames khớp chia `N`; reject event-count mismatch.
6. Implement `R@k = max(score[:k])`; quy định danh sách ngắn hơn `k` dùng toàn bộ danh sách, danh sách rỗng bằng 0.
7. Implement Final Score là mean năm cutoff; aggregate theo query là mean các Final Scores, nếu protocol sau này xác nhận khác thì thay reducer.
8. Tạo fixtures từ ví dụ PDF: KIS đúng/sai, Q&A đúng/sai, TRAKE `0.75`, Final Score `0.74`.
9. Tạo edge tests: boundary `s/e`, wrong video, duplicate response, empty list, >100 rows, non-monotonic TRAKE frames, Unicode answer.
10. Thiết kế dev split khóa trước tuning. Nếu BTC không cung cấp GT, tạo bộ query/interval nội bộ có provenance; cấm đánh giá trên chính query dùng để tune thủ công. Evaluator và fixtures chạy local, không cần full dataset; benchmark gắn video chạy trong Kaggle rồi chỉ export metric/report nhỏ.
11. Định nghĩa experiment record: data manifest hash, config, code revision, metrics, latency, memory, index size, notes.
12. Định nghĩa promotion comparison: delta theo cutoff/task; đánh dấu regressions ở R@1/R@5 thay vì chỉ aggregate.

## Success Criteria

- [x] Tests tái tạo toàn bộ ví dụ số trong tài liệu chính thức.
- [x] Boundary frames `s` và `e` đều được chấm đúng.
- [x] TRAKE partial credit và wrong-video zero hoạt động độc lập.
- [x] Submission >100 hoặc sai schema fail với thông báo actionable.
- [x] Report phân biệt official-formula score và semantic proxy.
- [x] Hai lần chạy cùng input cho output byte-stable hoặc numerically identical.
- [x] Promotion report hiển thị delta tại cả năm cutoff.

## Risk Assessment

- BTC chưa công bố semantic matcher/aggregate toàn bộ query. Giữ adapter và gắn nhãn proxy rõ ràng.
- Dev set nhỏ dễ overfit. Khóa split; giữ failure-case set; dùng paired per-query comparison.
- Tự động sửa submission có thể che lỗi. Validator chỉ đề xuất; không sửa ngầm.

## Rollback

Evaluator là pure computation. Quay lại scorer/config trước; rerun cùng immutable fixtures để xác nhận.
