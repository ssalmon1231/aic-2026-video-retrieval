---
phase: 9
title: "System Optimization and Competition Readiness"
status: pending
priority: P1
effort: "3-5 team-days plus rehearsal"
dependencies: [5, 7, 8]
optional_dependencies: [6]
---

# Phase 9: System Optimization and Competition Readiness

## Overview

Tích hợp Batch 2, đo bottleneck thật, freeze hệ thống và rehearsal cả ba task. Không thêm tính năng sau freeze trừ lỗi correctness hoặc benchmark có bằng chứng đủ lớn.

## Requirements

- Functional: ingest toàn bộ data release bằng pipeline đã kiểm thử.
- Functional: end-to-end workflow cho KIS, Q&A, TRAKE và final submission protocol.
- Functional: backup/restore index, config, drafts và export.
- Non-functional: offline-capable, reproducible, recoverable; resource budget đạt trên hardware thi.
- Non-functional: runbook rõ owner, command, expected output, fallback.

## Architecture

Giữ frozen bundle gồm code revision, environment lock, dataset manifest, active indexes, promoted configs, evaluator/submission adapter version và checksums. Bundle rehearsal là bundle dùng ngày thi.

## Related Code Files

- Create: `scripts/benchmark_system.py`
- Create: `scripts/verify_release.py`
- Create: `docs/competition-runbook.md`
- Create: `docs/submission-protocol.md` khi BTC công bố
- Create: `tests/test_end_to_end.py`
- Create: `config/competition.yaml`
- Create: `plans/260803-1208-aic2026-video-retrieval-system/reports/system-competition-readiness.md` trong lúc triển khai

## Implementation Steps

1. Re-run Phase 1 audit cho Batch 2; diff schema/artifact statistics; sửa tolerant ingestion, không fork pipeline nếu không bắt buộc.
2. Build/merge indexes bằng frozen model/preprocessing; verify row mapping, save/load probes và dataset fingerprints.
3. Chạy regression suite + full dev benchmark cho KIS/Q&A/TRAKE; so với frozen Phase 4/7/8 baselines và promoted Phase 6 config.
4. Profile end-to-end theo stage trên target hardware: startup, query encode, retrieval, rerank, decode, Q&A, TRAKE, UI/export; ghi p50/p95/peak resources.
5. Chỉ tối ưu top measured bottlenecks: cache bounded, batch/vectorize, narrow candidate/refinement depths hoặc approximate index qua promotion gate.
6. Khi BTC công bố submission protocol, implement đúng một adapter; thêm golden file/contract tests cho encoding, column order, rank, IDs và errors.
7. Tạo release verifier: checksums, package versions, model/index/config compatibility, free disk, writable draft/export paths, decoder smoke test.
8. Tạo backup: previous index/config, cold copy của manifest/checksums, export/draft recovery; test restore trên clean path.
9. Full rehearsal từ cold start: KIS query, automatic Q&A từ raw prompt đến ranked triples không human edits, TRAKE query, validator và export; đo tổng thời gian và ghi lỗi.
10. Chạy offline rehearsal bằng cách vô hiệu network; xác minh không component promoted nào treo hoặc cố tải model.
11. Freeze release bundle; cấm đổi model/index/config không kèm benchmark và rollback-ready artifact.
12. Hoàn thiện runbook ngày thi: startup, health check, task flows, backup operator, incident triage, fallback CLI, final submission checklist.

## Success Criteria

- [ ] Batch 1/2 cùng đi qua một manifest/index/retrieval contract.
- [ ] Full tests và benchmarks pass trên frozen bundle; không có regression chưa giải thích.
- [ ] p95 latency, peak RAM/GPU và disk nằm trong budget đã chốt.
- [ ] Offline cold-start hoàn tất ba task không tải dependency/model.
- [ ] Submission adapter qua golden contract tests; validator chấp nhận rehearsal export.
- [ ] Backup restore tạo hệ thống search được và drafts/exports đọc được.
- [ ] Runbook được một teammate khác thực hiện thành công.
- [ ] Release manifest có checksums và exact code/config/index identities.

## Risk Assessment

- Batch 2 đến muộn. Giữ incremental shards; audit/build commands resumable; ưu tiên correctness trước optional features.
- Protocol đổi sát hạn. Adapter isolation + golden tests giảm phạm vi sửa.
- Khác hardware làm vượt budget. Benchmark trên target sớm; có low-resource config đã rehearsal.
- Model cache gọi mạng lúc cold start. Vendor/cache model hợp lệ theo license; verify offline trước freeze.
- Last-minute tuning gây regression. Freeze gate; chỉ merge correctness fix với focused + full regression.

## Rollback

Giữ previous frozen bundle và active-index pointer. Khi release mới fail verifier/rehearsal, chuyển lại toàn bộ code/config/index bundle; không trộn version.
