# AIC 2026 Video Retrieval

Hệ thống retrieval chạy local-code/data-on-Kaggle cho vòng sơ tuyển AIC 2026.

Phạm vi hiện tại:

- Request 1 — Textual KIS: pipeline opt-in gồm raw Vietnamese retrieval, native-English holistic/clause/event routes, deterministic RRF, temporal ordering và private OCR exact-text retrieval.
- Request 2 — Automatic Q&A: baseline offline đã có parser, evidence windows, exact-frame sampling, strict answer protocol, fail-closed engine, pipeline API và CLI. VLM/T4 gate chưa chạy.
- Request 3 — TRAKE: baseline offline đã có parser, per-event retrieval adapter, same-video monotonic alignment, gap penalty, canonical frame IDs và CLI.

## Quy tắc dữ liệu

Dataset chính thức, video, CLIP arrays, index, model weights, private labels và Kaggle outputs **không được commit**. Chúng chỉ nằm trong Kaggle/private runtime.

Repo chỉ chứa source, tests, config mẫu, notebooks, docs và plans. Không commit `.env`, token, credential hoặc generated bundle.

## Cài đặt local

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash trên Windows
python -m pip install -e .
```

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python -m compileall src scripts
```

## Request 2/3 local baseline

Q&A nhận một raw prompt. Không có cờ nhập answer, frame hoặc rank:

```bash
python scripts/answer_query.py \
  --index /path/to/index.npz \
  --manifest /path/to/manifest.json \
  --dataset-root /path/to/dataset \
  --encoder-config /path/to/encoder.json \
  --config config/qa-baseline.yaml \
  --prompt "Mô tả cảnh. Câu hỏi: Người đó cầm gì?" \
  --output /path/to/qa-result.json
```

Không có VLM được promote, output Q&A rỗng theo fail-closed contract. UI có thể gọi `QaPipeline.answer_query(raw_text)`.

TRAKE nhận raw prompt chứa event theo thứ tự, phân cách bằng `|` hoặc `sau đó`:

```bash
python scripts/align_events.py \
  --index /path/to/index.npz \
  --encoder-config /path/to/encoder.json \
  --config config/trake-baseline.yaml \
  --prompt "Sự kiện một sau đó sự kiện hai" \
  --output /path/to/trake-result.json
```

UI có thể gọi `parse_trake_query()` và `retrieve_and_align()` trực tiếp. Cả hai baseline trả canonical `video_id` và zero-based `original_frame_id`.

## Rehearsal query pack và CSV nộp bài

Giữ query test, index, manifest, config encoder và output nộp bài trong private/Kaggle runtime. Runner đọc folder query có tên `*-kis.txt`, `*-qa.txt`, `*-trake.txt`, tạo ZIP chứa thư mục `submission/` và report aggregate không có raw prompt, answer hoặc frame ID:

```bash
python scripts/run_query_pack.py \
  --query-dir /private/THUNGHIEM-bo-de-thi \
  --index /private/index.npz \
  --manifest /private/manifest.json \
  --dataset-root /private/dataset \
  --encoder-config /private/encoder.json \
  --retrieval-config config/retrieval-baseline.yaml \
  --qa-config config/qa-baseline.yaml \
  --trake-config config/trake-baseline.yaml \
  --output /private/team-round.zip \
  --report /private/submission-report.json
```

Mỗi CSV là UTF-8, comma-separated, không header, tối đa 100 dòng. Q&A fail-closed tạo CSV rỗng nếu chưa có answer engine được promote. TRAKE đọc event dạng `E1: ... E2: ...` hoặc `sau đó`/`|`.

## Request 1 trên Kaggle

Runbook: [`docs/kaggle-runbook.md`](docs/kaggle-runbook.md)

Tạo source bundle local:

```bash
python scripts/prepare_kaggle_bundle.py
```

Bundle sinh tại `dist/`; thư mục này không thuộc Git.

## Hướng dẫn Git & Đẩy code lên GitHub

Khi làm việc trên nhánh tính năng hoặc chuẩn bị cập nhật code lên repository GitHub:

1. **Kiểm tra trạng thái các file:**
   ```bash
   git status
   ```
   *(Đảm bảo các file dataset, video, `.npz`, `.npy`, `*-bo-de-thi/`, `submission/` đều đã được `.gitignore` bảo vệ và không bị stage nhầm).*

2. **Stage và commit các thay đổi:**
   ```bash
   git add .
   git commit -m "feat: implement hybrid retrieval, QA baseline, TRAKE alignment, and evaluation tools"
   ```

3. **Push lên GitHub:**
   ```bash
   # Push nhánh hiện tại lên GitHub (ví dụ nhánh feature/request1-hybrid-retrieval)
   git push -u origin feature/request1-hybrid-retrieval

   # Hoặc nếu muốn đẩy lên master / main:
   # git checkout master
   # git merge feature/request1-hybrid-retrieval
   # git push origin master
   ```

## Tài liệu

- [Yêu cầu cuộc thi](docs/competition-requirements.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Kaggle runbook](docs/kaggle-runbook.md)
- [Toàn bộ implementation plan](plans/260803-1208-aic2026-video-retrieval-system/plan.md)

## Trạng thái

Request 1 có hybrid baseline nhưng chưa có locked private held-out metrics. Request 2/3 có offline core usable cho UI integration; real VLM answer quality, T4 compatibility, trial-web integration và competition accuracy chưa xác minh.

