# AIC 2026 Video Retrieval

Hệ thống retrieval chạy local-code/data-on-Kaggle cho vòng sơ tuyển AIC 2026.

Phạm vi hiện tại:

- Request 1 — Textual KIS: exact NumPy index, English/Vietnamese query encoder candidates, deterministic ranking và evaluator.
- Request 2 — Automatic Q&A: đang triển khai pipeline prompt-only, tự retrieve, localize, answer, verify và rank.
- Request 3 — TRAKE: thiết kế sau khi Request 1–2 có baseline dùng được.

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

Optional extras:

```bash
python -m pip install -e ".[video]"
python -m pip install -e ".[clip]"
python -m pip install -e ".[multilingual]"
```

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python -m compileall src scripts
```

## Request 1 trên Kaggle

Runbook: [`docs/kaggle-runbook.md`](docs/kaggle-runbook.md)

Tạo source bundle local:

```bash
python scripts/prepare_kaggle_bundle.py
```

Bundle sinh tại `dist/`; thư mục này không thuộc Git.

## Request 2 — phân chia công việc

Thiết kế chính: [`plans/260803-1208-aic2026-video-retrieval-system/phase-07-q-a-pipeline.md`](plans/260803-1208-aic2026-video-retrieval-system/phase-07-q-a-pipeline.md)

Thứ tự triển khai:

1. **7A — Contracts/query routing:** frozen Q&A contracts, parser/router Việt–Anh, no-human public API.
2. **7B — Retrieval/evidence:** multi-variant exact retrieval, temporal clustering, exact-frame windows/sampling.
3. **7C — T4 model gate:** so sánh VLM candidates trên Kaggle NVIDIA T4; chưa pin production model trước gate.
4. **7D–7H — Integration:** answer engine, verifier/ranking, optional OCR/ASR, CLI/notebook và benchmark.

Nhánh gợi ý:

```text
feature/request2-7a-contracts
feature/request2-7b-evidence
experiment/request2-7c-vlm-gate
```

Không sửa cùng file trên hai nhánh nếu chưa thống nhất ownership. Mỗi pull request cần:

- scope nhỏ, không kèm dataset/output;
- tests cho behavior mới;
- `python -m unittest discover -s tests -v` pass;
- `python -m compileall src scripts` pass;
- mô tả contract/public API thay đổi nếu có.

## Tài liệu

- [Yêu cầu cuộc thi](docs/competition-requirements.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Kaggle runbook](docs/kaggle-runbook.md)
- [Toàn bộ implementation plan](plans/260803-1208-aic2026-video-retrieval-system/plan.md)

## Trạng thái

Request 1 baseline code và Kaggle workflow đã có. Request 2 Phase 7A–7B là mốc code tiếp theo. Dataset audit/index/model gates thật vẫn chạy trên Kaggle.
