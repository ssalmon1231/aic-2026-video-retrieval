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

## 7. Reusable Request 1 hybrid runtime

Interactive notebook load exact index, Vietnamese encoder, native English encoder và planner đúng một lần, sau đó gọi `search_query(text)` cho nhiều query. Không chạy audit, rebuild index, reload model hoặc spawn `scripts/search.py` cho mỗi query.

`config/hybrid-retrieval.example.yaml` committed với `"enabled": false`. Disabled CLI path chỉ load raw Vietnamese encoder và trả exact B0; không load English encoder, planner hoặc OCR artifact. Tạo runtime copy dưới `/kaggle/working` rồi bật riêng cho private experiment.

Hybrid flow:

1. raw Vietnamese CLIP list;
2. bounded English holistic/clause/event lists;
3. weighted RRF candidate union;
4. monotonic same-video temporal boost cho ordered events;
5. optional private OCR exact/fuzzy evidence.

Planner strict JSON, deterministic `do_sample=False`, bounded 1 holistic, 4 clauses, 3 ordered events và 4 exact-text strings. Raw query, generated plan và OCR strings chỉ tồn tại trong private process/artifact. Output chỉ ghi aggregate flags/counts/timings.

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/search.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --query-text "<private-query>" \
  --encoder-config /kaggle/working/query-encoder-vietnamese.json \
  --english-encoder-config /kaggle/working/query-encoder-english.json \
  --hybrid-config /kaggle/working/hybrid-retrieval.json \
  --ocr-artifact /kaggle/input/private-ocr/ocr-artifact \
  --config config/retrieval-baseline.yaml \
  --output /kaggle/working/aic-results/kis-hybrid.json
```

`--ocr-artifact` optional. Missing artifact neutral. Corrupt/checksum/manifest/index mismatch fail closed before retrieval. Auxiliary runtime failure returns exact raw B0 responses.

### Build private OCR artifact

Chỉ build trên official keyframes/private Kaggle T4. PaddleOCR không thuộc core dependency vì package/API/model compatibility chưa được xác minh trên T4.

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/build_ocr_artifact.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --dataset-root /kaggle/input/<official-dataset> \
  --output /kaggle/working/ocr-artifact \
  --device gpu:0 \
  --model-revision <verified-model-revision>
```

Builder mặc định thử `PP-OCRv5_mobile_det` + `latin_PP-OCRv5_mobile_rec`. Trước full build phải smoke exact installed PaddleOCR version, constructor/predict schema, model revision/hash, Vietnamese diacritics, throughput, VRAM và batch size trên T4. Artifact gồm descriptor + JSONL OCR records; giữ private, không commit hoặc public.

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

Vector B0 command phía trên vẫn dùng được. Full raw-text B0–B3 benchmark dùng private ID→text mapping riêng:

```bash
PYTHONPATH=/kaggle/working/aic-retrieval/src python scripts/benchmark_kis.py \
  --index /kaggle/working/aic-index/clip-flat-ip.npz \
  --private-query-texts /kaggle/working/private/query-texts.json \
  --encoder-config /kaggle/working/query-encoder-vietnamese.json \
  --english-encoder-config /kaggle/working/query-encoder-english.json \
  --hybrid-config /kaggle/working/hybrid-retrieval.json \
  --ocr-artifact /kaggle/input/private-ocr/ocr-artifact \
  --query-set /kaggle/working/private/kis-queries.json \
  --retrieval-config config/retrieval-baseline.yaml \
  --name request1-b3 \
  --code-revision <git-revision-or-source-sha256> \
  --output /kaggle/working/aic-results/kis-benchmark-b3.json
```

Private mapping schema:

```json
{"version":1,"queries":{"query-id":"raw private text"}}
```

IDs phải khớp chính xác query set. Runner public chỉ nhận query ID; aggregate report không chứa raw text, generated plan, OCR strings hoặc ground truth.

Đo tuần tự trên cùng locked split:

- B0 raw multilingual baseline;
- B1 native-English holistic/clauses + RRF;
- B2 ordered temporal alignment;
- B3 private OCR exact-text fusion.

Sau warmup ghi R@1/R@5/R@20/R@50/R@100, Final Score, p50/p95, candidate recall, fallback count, peak GPU/host memory. Dimension khớp không chứng minh encoder compatibility. Chỉ locked private held-out evidence được promote; R@1/R@5 guardrails, T4 no-OOM và response identity checks phải pass. Contact sheet chỉ smoke path.

## 9. Artifacts cần giữ

- `aic-diagnostic/report.json`
- `aic-audit/report.json`
- `aic-audit/manifest.json`
- verified index metadata, archive size
- `kis-benchmark-b0.json` đến `kis-benchmark-b3.json`
- private `ocr-artifact/` nếu B3 được chạy
- source bundle SHA-256 và code revision

Không export sample frames, official/private query labels, raw query vectors/texts, generated plans, OCR strings, credentials hoặc private metadata vào git/public output.
