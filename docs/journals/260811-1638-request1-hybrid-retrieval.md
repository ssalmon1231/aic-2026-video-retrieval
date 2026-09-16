---
title: "Request 1 Hybrid Retrieval Local Completion"
date: "2026-08-11"
status: completed-local
branch: "feature/request1-hybrid-retrieval"
tags: [request-1, hybrid-retrieval, temporal, ocr, kaggle]
---

# Request 1 Hybrid Retrieval Local Completion

## Context

Request 1 cần xử lý query Việt dài, ordered events và exact visible text trước khi tiếp tục Request 2.

## What happened

- Thêm bounded private query planner.
- Thêm raw Vietnamese + native-English multi-list retrieval và weighted RRF.
- Thêm same-video monotonic temporal alignment.
- Thêm private OCR artifact, exact/fuzzy lookup và OCR-only candidate recovery.
- Tích hợp CLI, private raw-text benchmark, Kaggle notebook và source bundle.
- Giữ exact raw baseline cho disabled mode và auxiliary failures.
- Giữ public outputs không chứa raw query, generated plan, OCR strings hoặc ground truth.

## Verification

- Full local suite: `159` tests pass.
- `compileall`, Python 3.10 AST (`42` files, `2` notebooks), `git diff --check`: pass.
- Deterministic source bundle: `55` files, SHA-256 `11298390fd22bcec743f2cecb3091d6ea1c160ba07980b0a89d03864063f5e5c`.

## Decisions

- Hybrid config mặc định disabled.
- PaddleOCR không vào core dependency trước T4 compatibility gate.
- Không claim accuracy improvement từ synthetic tests hoặc contact sheets.
- Request 2 Phase 7B–7C tiếp tục tạm dừng.

## Next

1. Chạy locked private B0–B3 benchmark trên Kaggle NVIDIA T4.
2. Xác minh PaddleOCR package/model revision, VRAM, latency và Vietnamese diacritics.
3. Promote chỉ khi held-out Final Score tăng và R@1/R@5 guardrails pass.
