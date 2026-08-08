# Kaggle Runbook

## 1. Chuẩn bị source bundle local

```bash
python scripts/prepare_kaggle_bundle.py
```

Artifact duy nhất cần upload:

```text
dist/aic-retrieval-kaggle.zip
```

Lưu SHA-256 do lệnh in ra. Không thêm PDF, dataset, video, `.npy`, `.npz`, query labels, credentials hoặc private metadata vào ZIP.

## 2. Chuẩn bị Kaggle runtime

1. Upload ZIP thành private Kaggle Dataset/code artifact.
2. Tạo notebook; chọn **Accelerator = GPU T4**, bật Internet, attach official AIC dataset và source bundle. Không dùng P100 khi chưa kiểm tra compatibility.
3. Kiểm tra mount thực tế dưới `/kaggle/input`; không đoán dataset slug.
4. Copy source đã được Kaggle tự giải nén sang writable `/kaggle/working/aic-retrieval`. Thư mục nguồn đúng phải chứa `pyproject.toml` và `scripts/audit_dataset.py`.
5. Chạy trực tiếp từ source; không dùng `pip install` khi notebook không có Internet:

```python
import os
os.environ["PYTHONPATH"] = "/kaggle/working/aic-retrieval/src"
```

Các lệnh shell bên dưới cần prefix:

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python ...
```

Không sửa file trong `/kaggle/input`. Tạo config runtime trong `/kaggle/working` từ các example JSON-compatible YAML.

## 3. Evidence diagnostic đã xác nhận

Diagnostic trên dataset hiện tại xác nhận:

- 873 video/mapping/CLIP/metadata; 177.321 keyframe/object;
- `round(pts_time * fps)` thắng 256/256 hàng sampled và được dùng làm canonical frame ID;
- object JSON dùng năm parallel arrays; score chuỗi số được parser xác thực;
- 192 video có CSV `frame_idx` lặp nhưng mọi CLIP row/keyframe vẫn được giữ.

Không cần chạy lại diagnostic nếu dataset/source revision không đổi. Vẫn cần xác nhận artifact `aic25-b1` được phép dùng cho AIC2026.

## 4. Audit dataset bằng adapter đã xác minh

Upload source bundle mới. Copy `config/dataset.example.yaml` thành `/kaggle/working/dataset.json`; giữ `"frame_id_source": "timestamp"`. Chạy:

```bash
cd /kaggle/working/aic-retrieval
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/audit_dataset.py \
  --config /kaggle/working/dataset.json
```

Chỉ tiếp tục khi exit code `0`, report có `"valid": true`, manifest tồn tại. Nếu schema/path khác config: giữ report lỗi, cập nhật parser/config local; không sửa hoặc di chuyển source dataset.

## 5. Build và verify exact index

Trước build, xác minh tên model/revision và preprocessing của provided CLIP features. Không dùng giá trị `unverified` như provenance thật.

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/build_index.py \
  --config /kaggle/working/index.json
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/verify_index.py \
  /kaggle/working/aic-index/clip-flat-ip.npz
```

Giữ NumPy exact index đến khi resource profile chứng minh không đạt budget.

## 6. Xác minh raw-text KIS tiếng Anh và tiếng Việt

Cài từng optional encoder dependency khi Kaggle Internet đang bật:

```bash
pip install -e '/kaggle/working/aic-retrieval[clip]'
pip install -e '/kaggle/working/aic-retrieval[multilingual]'
```

Hai config được pin độc lập:

- `config/query-encoder.example.yaml`: English control `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`;
- `config/query-encoder-multilingual.example.yaml`: Vietnamese candidate `sentence-transformers/clip-ViT-B-32-multilingual-v1@58edf8cada9e398793dca955574a48cbb7f18be2`, Apache-2.0, model card công bố 50+ ngôn ngữ gồm `vi` và alignment với `clip-ViT-B-32` image space.

OpenAI revision cung cấp PyTorch weights, không có `model.safetensors`; loader giữ `use_safetensors=False`. Cả hai backend load exact commit, đặt `trust_remote_code=False` cho Sentence Transformers, kiểm tra `_commit_hash` khi runtime công bố field đó. Không cần `HF_TOKEN`; không ghi token hoặc Hugging Face cache vào source bundle.

Tạo hai runtime config dưới `/kaggle/working`, rồi chạy cùng exact index/config:

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/search.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --query-text "a man riding a motorcycle on a street" \
  --encoder-config /kaggle/working/query-encoder.json \
  --config config/retrieval-baseline.yaml \
  --output /kaggle/working/aic-results/kis-search-english.json

PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/search.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --query-text "một người đàn ông đang chạy xe máy trên đường phố" \
  --encoder-config /kaggle/working/query-encoder-multilingual.json \
  --config config/retrieval-baseline.yaml \
  --output /kaggle/working/aic-results/kis-search-vietnamese.json
```

Output ghi backend/model/revision/runtime provenance, `encoding_elapsed_ms`, `retrieval_elapsed_ms`; không serialize raw query. `elapsed_ms` giữ tổng tương thích ngược. Không bỏ dấu, sửa negation hoặc tự dịch. Model card alignment và cùng dimension chưa chứng minh compatibility với provided AIC vectors; chỉ private labeled R@k cho phép promote backend multilingual.

Vector mode cũ vẫn độc lập, không load `torch`/`transformers`/`sentence-transformers`:

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/search.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --query-vector /kaggle/working/private/query-vector.npy \
  --config config/retrieval-baseline.yaml \
  --output /kaggle/working/aic-results/kis-vector-search.json
```

## 7. Reusable query runtime và optional reranker

Interactive notebook phải load exact index và `QueryEncoderRuntime` một lần, sau đó gọi `search_query(text)` cho nhiều query. Không chạy audit, rebuild index, reload model hoặc spawn `scripts/search.py` cho mỗi query.

`config/reranker-disabled.example.yaml` là mặc định committed. Khi disabled, output phải giống exact baseline. Chỉ bật trong private experiment sau khi tạo runtime config dưới `/kaggle/working` với pinned Qwen revision và explicit nonzero weights từ dev split; không commit weights chưa được benchmark.

Reranker chỉ xử lý exact top-500. Planner warm sinh một positive, tối đa ba negatives và bounded object labels; generated content chỉ tồn tại trong memory. Deterministic generation dùng compact JSON contract và cap `96` generated tokens. Notebook/output JSON chỉ được ghi trạng thái aggregate `applied`, `fallback`, `circuit_open`, `planner_elapsed_ms`, `scoring_elapsed_ms` cùng bounded generated-token counts; không print/serialize raw query, prompt, generated plan, object labels/paths/detections.

Trước smoke query, warm planner hai lần ngoài `ContrastiveReranker`: call đầu khởi động model/CUDA, call sau dùng fixed public synthetic visual query tiếng Việt với cùng empty vocabulary của Contrastive Text-only. Discard cả hai plans; chỉ tạo reranker sau khi cả hai parse thành công. Warm-up timing/token counts là startup diagnostics, không thay thế post-warm-up latency gate.

Planner load/generation/scoring failure trả exact baseline. Load/warm-up failure đánh dấu aggregate `fallback=true`, `circuit_open=true`; runtime giữ baseline đến khi restart. Planner vượt 1,5 giây trả baseline cho query hiện tại, mở in-memory circuit breaker; query sau giữ baseline đến khi restart runtime. Không hard-cancel CUDA generation bằng thread timeout.

## 8. Benchmark KIS

Chỉ chạy khi có đủ:

- text encoder/tokenizer/preprocessing đã pin và tương thích image features;
- private labeled KIS query set;
- ordered 2D query-vector `.npy` cùng thứ tự query IDs cho vector baseline, hoặc private in-memory runner cho full raw-text pipeline;
- verified index.

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/benchmark_kis.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --query-vectors /kaggle/working/private/query-vectors.npy \
  --query-set /kaggle/working/private/kis-queries.json \
  --retrieval-config config/retrieval-baseline.yaml \
  --name kaggle-baseline \
  --code-revision <git-revision-or-source-sha256> \
  --output /kaggle/working/aic-results/kis-benchmark.json
```

Đo bốn ablations trên cùng locked split: baseline, object-only, contrastive-only, object + contrastive. Full-pipeline timer bao gồm text encoding, exact search, planner, reranking và dedup. Sau warmup ghi p50/p95, startup, planner/scoring timing, peak GPU/host memory. Target T4 chưa xác minh: base encode + exact search `< 500 ms`, object `< 100 ms`, contrastive `< 200 ms`, planner `< 1,5 s`, total p95 `< 2 s`. Một smoke query đạt gate chỉ xác minh notebook path; promotion vẫn cần locked private multi-query p50/p95 và R@k/Final Score.

Dimension khớp không chứng minh encoder tương thích. Chỉ private held-out R@k/Final Score được promote encoder/reranker. Runner chỉ nhận query ID; aggregate report không chứa raw query, plan hoặc ground-truth content.

## 9. Artifacts cần giữ

- `aic-diagnostic/report.json`
- `aic-audit/report.json`
- `aic-audit/manifest.json`
- verified index metadata, archive size
- `kis-benchmark.json`
- source bundle SHA-256 và code revision

Không export sample frames, official/private query labels, raw query vectors, credentials hoặc private metadata vào git/public output.
