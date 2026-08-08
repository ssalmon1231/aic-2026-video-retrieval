# AIC 2026 Evaluation Protocol

## Scope

Evaluator tái tạo phần công thức đã công bố. Mỗi query nhận tối đa 100 ranked responses.

- Textual KIS: đúng video và frame thuộc inclusive interval `[s,e]`.
- Q&A: điều kiện KIS cộng answer matcher.
- TRAKE: sai video bằng `0`; đúng video nhận tỷ lệ event frames thuộc từng interval tương ứng.
- `R@k = max(response_scores[:k])`, với `k ∈ {1,5,20,50,100}`.
- Query Final Score là mean của năm `R@k`; danh sách ngắn dùng toàn bộ responses, danh sách rỗng bằng `0`.
- Aggregate local hiện là mean Query Final Score. Thay reducer nếu BTC công bố protocol khác.

## Q&A matcher labels

- `exact`: Unicode-preserving `casefold` cộng whitespace normalization, sau đó exact equality với một answer hợp lệ.
- `local-semantic-proxy`: matcher do dev set định nghĩa; report phải giữ đúng nhãn này.
- `official`: chưa có implementation. Không báo proxy như official score.

Raw answer trong response không bị sửa.

## Validation

Evaluator fail thay vì tự sửa khi:

- Sai type/schema.
- Hơn 100 responses.
- Duplicate response.
- Frame ID âm.
- TRAKE event count không khớp.
- TRAKE frames không strictly increasing.

## Experiment records

Mỗi experiment lưu:

- Manifest SHA-256.
- Code revision.
- Config.
- Per-query metrics, mean `R@1/R@5/R@20/R@50/R@100`, Final Score.
- p50/p95 latency, peak memory, index size.
- Notes.

JSON dùng sorted keys và atomic replacement. Promotion comparison luôn hiển thị delta tại cả năm cutoff; không promote chỉ vì aggregate tăng nếu R@1/R@5 regression vượt gate đã định.

## Reranker ablations

Reranker mặc định tắt. Mọi candidate phải dùng cùng exact shortlist, encoder revision, query IDs và deterministic planner. Chạy bốn ablations:

1. Baseline.
2. Object-only.
3. Contrastive-only.
4. Object + contrastive.

Timing bắt đầu trước raw-text encoding và kết thúc sau reranking/dedup. Report p50/p95 toàn query; ghi riêng startup, planner, object/contrastive scoring và peak GPU/host memory. Target Kaggle T4 sau warmup: base encode + exact search `< 500 ms`, object scoring `< 100 ms`, contrastive scoring `< 200 ms`, planner `< 1.5 s`, total p95 `< 2 s`. Target là gate cần đo, không phải kết quả đã xác minh.

## Dev split và promotion

Khóa query IDs trước tuning. Tune weights chỉ trên private development split; evaluate held-out một lần. Không dùng query đã tune thủ công làm held-out evaluation. Promote chỉ khi held-out Final Score tăng, không có R@1/R@5 regression vượt gate, latency/memory nằm trong T4 budget và artifact tái tạo được.

Benchmark runner chỉ nhận `query_id`; private closure giữ raw query trong memory. Report không chứa raw query, generated plan, required/excluded labels, ground truth content, detections, object paths hoặc query vectors. Dataset-bound benchmark chạy trên Kaggle; chỉ export aggregate report nhỏ, không export frames/private metadata. Planner failure hoặc quá budget phải khớp exact baseline cho query đó; quá budget mở circuit breaker tới khi restart runtime.
