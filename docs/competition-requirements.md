# AIC 2026 Competition Requirements

## Source hierarchy

1. `Thong tin vong So tuyen AIC2026.pdf`: task/output/metric contract.
2. Ba tài liệu tập huấn: implementation guidance, không thay contract.
3. Repo/paper ngoài: prior art, chỉ dùng sau license/benchmark gate.

## Task contract

Mỗi query trả tối đa 100 responses, rank từ tốt nhất xuống.

| Request | Response | R-Score |
|---|---|---|
| Textual KIS | `video_id, frame_id` | `1` khi đúng video và `frame_id ∈ [s,e]`; ngược lại `0` |
| Q&A | `video_id, frame_id, answer` | `1` khi điều kiện KIS đúng và answer khớp ngữ nghĩa; ngược lại `0` |
| TRAKE | `video_id, frame_id_1, …, frame_id_N` | Sai video: `0`; đúng video: tỷ lệ event frames nằm trong interval tương ứng |

Với `k ∈ {1,5,20,50,100}`:

```text
R@k = max(R-Score(response_i)), 1 ≤ i ≤ k
Final Score = mean(R@1, R@5, R@20, R@50, R@100)
```

Semantic-answer matcher và global aggregation ngoài công thức trên chưa được công bố đầy đủ. Local proxy phải gắn nhãn `local-semantic-proxy`, không báo như official score.

## Data contract

- Video chính thức là source of truth.
- Semantic keyframe không phải compression I-frame.
- Mọi output dùng original frame ID.
- Canonical video fields: `video_id`, `video_path`, `fps`, `frame_count`, `duration`, `batch_id`, nullable `metadata_path`.
- Canonical keyframe fields: `video_id`, `keyframe_id`, `keyframe_path`, `keyframe_ordinal`, `original_frame_id`, nullable `clip_row`, nullable `object_path`.
- `video_id + keyframe_id`, per-video ordinal và per-video CLIP row phải duy nhất.
- `original_frame_id` phải thuộc video bounds; mapping phải tăng theo timeline.
- CLIP array phải là ma trận số finite; số row phải khớp mapping.
- Metadata, object JSON, CLIP feature là support artifacts. Thiếu metadata/object tạo warning; mapping sai làm audit fail.

## Runtime contract

- Dataset/index lớn chỉ nằm trên Kaggle.
- Audit đọc source data; không mutate.
- Manifest/report xuất vào `/kaggle/working` bằng atomic replacement.
- Local chỉ giữ code, config, fixtures và artifacts nhỏ.
- Không ghi sample frames, credentials hoặc private metadata vào report/git.

## Phase 1 verification state

Đã xác minh local:

- Python package tối thiểu: Python `>=3.10`, NumPy; OpenCV là optional extra cho video/frame verification.
- Audit read-only kiểm tra IDs, video probe, mapping, CLIP shape/row/dimension/finite/unit norm, support JSON và sampled exact-frame MAE.
- Invalid audit xóa manifest cũ; report/manifest hợp lệ được publish atomically.
- Prior art hiện chỉ ở mức ADOPT concept/algorithm; chưa ADAPT source code ngoài, nên chưa phát sinh dependency hoặc license lineage mới.

Chỉ chốt trên Kaggle sau khi dataset/runtime chính thức sẵn sàng:

- Python/GPU/CUDA/RAM/disk matrix.
- CLIP tokenizer/model/preprocessing identity và empirical image-text similarity; chưa đủ bằng chứng thì `recompute-required`.
- 100% inventory và sampled đầu/giữa/cuối frame mapping trên từng batch.

## Current delivery order

`Phase 1 → Phase 2/3 → Phase 4 → Phase 5 KIS tối thiểu → Phase 7 Q&A`.
TRAKE triển khai sau baseline Request 1–2; contracts/evaluator vẫn bao phủ ngay từ đầu.

## Unresolved questions

- Batch 2 schema và ngày công bố.
- Submission transport/schema cuối.
- Latency, query count, submission limits.
- Official semantic-answer matcher.
- Internet/model/API policy và Kaggle hardware khi thi.
