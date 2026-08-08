---
title: Session recovery checkpoint
status: completed
priority: P1
effort: small
branch: master
tags: [recovery, verification, kaggle]
created: 2026-08-05
---

# Session recovery checkpoint

Status: local recovery verification completed
Date: 2026-08-05
Scope: AIC 2026 Request 1–2 baseline plus optional shortlist reranker

## Durable state

- Source changes are saved on disk in this repository.
- No commit, push, upload, Kaggle mutation, private dataset read, or PDF mutation occurred.
- Repository content is largely untracked. Do not assume `git diff` alone captures every file.
- Reranker remains disabled by default.
- Notebook bundle checksum is pinned to `dc9ae93704ab9adb15c87ab0ebdf466604d29fb51d7ca45f6b9601e78dc8d0e3` after deterministic verification.

## Implemented

- Exact top-500 shortlist linkage and stable reranking.
- Warm multilingual text encoder with batched encoding.
- Strict pinned Qwen planner and bounded internal contrastive plan.
- Object-aware shortlist scoring.
- Exact-baseline fallback for generation, parsing, scoring, timeout, and planner startup failure.
- Aggregate private-safe reranker status only.
- Query-ID benchmark runner.
- Mock-only tests, notebook workflow, docs, deterministic bundle builder.

Latest startup-fallback changes:

- `scripts/search.py`: planner constructor failure preserves exact baseline; reports `fallback=true`, `circuit_open=true` without exception/query text.
- `tests/test_retrieval.py`: regression test for planner constructor failure and private-safe exact baseline.
- `notebooks/aic-kis-kaggle.ipynb`, cell 20: planner load failure no longer stops notebook.
- `docs/kaggle-runbook.md`: planner load fallback documented.

## Recovery result

Completed locally on Python 3.14.6:

- Fresh shell runner works; `python -V` no longer returns `ENAMETOOLONG`.
- Focused suite: 69 tests passed.
- Full suite: 96 tests passed.
- `compileall`, `git diff --check`, Python 3.10 AST parse: passed.
- Deterministic Kaggle bundle: 41 sorted files; identical bytes across two builds.
- Bundle SHA-256: `dc9ae93704ab9adb15c87ab0ebdf466604d29fb51d7ca45f6b9601e78dc8d0e3`.
- Both archives clean for forbidden datasets, vectors, queries, labels, credentials, model/cache files, PDFs, media, nested archives.
- Notebook checksum pinned; reranker remains disabled.
- One Python 3.14-only test assertion issue fixed with explicit NumPy array comparison; runtime behavior unchanged.

Remaining environment issue: delegated tester/debugger worktrees still fail before execution:

```text
WorktreeCreate hook failed: hook succeeded but returned no worktree path
```

This does not block direct local verification. Diagnose Claude Terminal worktree hook separately if subagent isolation is needed.

## Resume sequence

1. Start a fresh Claude session in `E:\15.DEEPLEARNING\Aichallenge2`; do not paste the old transcript or long summaries.
2. Read this file, relevant project docs, current task list, and only files needed for the next step.
3. Smoke-test the fresh runner:

```bash
python -V
```

4. If the smoke test still returns `ENAMETOOLONG`, restart Claude Terminal completely. If it persists after restart, diagnose the app/tool runner separately before changing project code.
5. Run focused tests:

```bash
python -m unittest tests.test_query_encoding tests.test_dataset_audit tests.test_reranking tests.test_retrieval tests.test_kis_benchmark tests.test_kaggle_bundle -v
```

6. Fix only confirmed failures. Then run:

```bash
python -m unittest discover -s tests -v
python -m compileall src scripts
git diff --check
```

7. Parse all Python source and every notebook code cell with Python 3.10-compatible `ast.parse`.
8. Build the Kaggle ZIP twice. Require identical sorted entries and SHA-256. Scan both archives for datasets, vectors, queries, labels, credentials, model/cache files, PDFs, media, and nested archives.
9. Replace `__BUNDLE_SHA256_PENDING__` only after deterministic verification.
10. Keep `ENABLE_RERANKING = False` until private Kaggle T4 ablations improve held-out retrieval metrics within latency/memory gates.

## Safety constraints

- Do not read Claude settings files or repeat previously exposed tokens. Rotate those tokens before reuse.
- Do not access or copy the private dataset locally.
- Do not modify, move, or delete the four competition PDFs.
- Do not commit PDFs, datasets, model weights, caches, query vectors, labels, raw queries, detections, credentials, or private metadata.
- Do not upload, commit, push, or mutate Kaggle state without explicit approval.
- Default GPU: NVIDIA T4. Do not recommend P100 without compatibility evidence.

## Fresh-session prompt

```text
Đọc plans/260805-2207-session-recovery/plan.md và các docs liên quan. Tiếp tục từ mục Resume sequence, bắt đầu bằng `python -V`. Không đọc transcript cũ trừ khi handoff thiếu dữ kiện. Không hỏi lại quyết định đã ghi. Không upload/commit/push.
```
