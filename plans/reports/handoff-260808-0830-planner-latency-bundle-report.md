# Handoff: AIC 2026 planner-latency bundle

## Mission

Hoàn tất task duy nhất còn mở:

`#15 [in_progress] Build và xác minh bundle`

Không làm lại task đã hoàn tất:

- `#12 [completed] Giảm Qwen decode workload`
- `#13 [completed] Cập nhật Contrastive notebook`
- `#14 [completed] Cập nhật Kaggle runbook`

Kế hoạch đã duyệt, authoritative:

`C:\Users\minht\.claude\plans\splendid-fluttering-swing.md`

## Quyết định đã chốt

- Giữ `planner_budget_seconds=1.5`.
- Giữ `MAX_NEW_TOKENS=96`.
- Generated-token count hợp lệ phải thỏa `0 < count < 96`; count `96` có thể bị truncate nên smoke evidence không hợp lệ.
- Giữ tối đa ba negatives.
- Giữ compact planner prompt, strict four-key JSON schema, pinned Qwen model/revision, FP16 CUDA và `do_sample=False`.
- Warm planner trực tiếp hai lần ngoài `ContrastiveReranker`:
  1. English public synthetic query để warm model/CUDA kernel.
  2. Vietnamese public synthetic query đại diện.
- Discard cả hai warm-up plans. Chỉ tạo reranker khi cả hai parse thành công.
- Chỉ serialize aggregate telemetry: elapsed time và generated-token count. Không output prompt, generated plan, raw private query, vectors, object metadata, credentials hoặc frames.
- Giữ exact-baseline fallback và fail-closed in-memory circuit breaker.
- Không tăng cap hoặc latency budget chỉ để pass.
- Không sửa baseline notebook.
- Không ghi đè artifact cũ.
- Promotion vẫn blocked tới khi có locked private multi-query post-warm-up p50/p95 và labeled R@k/Final Score.

## Project state

Project root:

`E:\15.DEEPLEARNING\Aichallenge2`

Branch: `master`. Repository bắt đầu với gần như toàn bộ project files untracked; Git history không chứng minh byte identity. Không commit, push hoặc upload.

Đã sửa và cần verify:

- `src/aic_retrieval/reranking.py`
- `tests/test_reranking.py`
- `notebooks/aic-kis-kaggle-contrastive-only.ipynb`
- `docs/kaggle-runbook.md`

Phải giữ nguyên:

- `notebooks/aic-kis-kaggle.ipynb`
- `config/reranker-disabled.example.yaml`
- `docs/evaluation-protocol.md`
- `dist/aic-retrieval-kaggle-contrastive-fix.zip`

Artifact cần tạo:

`dist/aic-retrieval-kaggle-planner-latency.zip`

Contrastive notebook vẫn có placeholder chưa pin:

```python
EXPECTED_BUNDLE_SHA256 = "__PLANNER_LATENCY_BUNDLE_SHA256__"
```

## Code state quan trọng

### Planner

`src/aic_retrieval/reranking.py` hiện có:

- `MAX_NEW_TOKENS = 96`
- `MAX_NEGATIVES = 3`
- compact JSON prompt
- `QwenPlanner.last_generated_tokens`
- reset token count trước mỗi call
- token count set ngay sau `generate()` và trước decode/parse
- sanitized `RerankingError("planner generation failed")`
- unchanged `planner_budget_seconds=1.5` timing boundary
- exact fallback và circuit breaker nguyên vẹn

### Contrastive notebook

`notebooks/aic-kis-kaggle-contrastive-only.ipynb` hiện có:

```python
PLANNER_WARMUP_QUERIES = (
    "a single person standing outdoors",
    "một chiếc ô tô màu đỏ đỗ bên đường",
)
```

Hai warm-ups gọi `planner.plan(warmup_query, ())` trực tiếp. Reranker chỉ được tạo nếu cả hai parse thành công.

Warm-up gate và real-query gate dùng strict:

```python
0 < generated_tokens < MAX_NEW_TOKENS
```

`search_query()` capture `planner_attempted` trước retrieval để không tái dùng stale warm-up count khi circuit đã mở hoặc planner không chạy. Khi planner không được gọi, telemetry query thật là `0`.

`experiment_valid` yêu cầu:

- reranking enabled
- cả hai warm-ups thành công
- warm-up aggregate telemetry hợp lệ
- real query token count hợp lệ
- `planner_elapsed_ms < 1500`
- `applied is True`
- `fallback is False`
- `circuit_open is False`

Notebook edit cuối chưa có fresh local AST/test evidence.

## Blocker hiện tại: Bash executor

Mọi Bash tool và local `!` command fail trước command body:

```text
/usr/bin/bash: line 192: expo: command not found
```

Fresh proof trong session cũ:

```text
Bash command: true
Exit code 127
/usr/bin/bash: line 192: expo: command not found
```

Command body marker cũng không chạy:

```bash
printf '%s\n' 'COMMAND_BODY_REACHED' >&2; exit 0
```

Output chỉ có cùng line-192 error. Vì vậy failure nằm trong prepended/generated startup composition, trước requested command body.

Evidence đã loại trừ:

- Current snapshot `C:\Users\minht\.claude\shell-snapshots\snapshot-bash-1786151805199-7e7jn3.sh` chỉ 45 dòng và không có standalone `expo`.
- Exact token searches không tìm thấy standalone `expo` trong current shell snapshot, known `session-env/**/*.sh`, user hooks hoặc Claude Terminal hook handler.
- Claude Terminal hook handler chỉ forward hook JSON qua localhost; không mutate Bash command.
- `dangerouslyDisableSandbox=true`, clean `PATH`, local `!` command và restored old snapshot đều không sửa lỗi.
- Không được sửa profile, snapshot, hook hoặc settings khi chưa locate producer/source thật.

Related runner failures:

- Agent launch fail vì `WorktreeCreate` hook báo success nhưng không trả worktree path.
- Claude Terminal terminal/tab renderer từng timeout.
- Temporary workflows và quick action không tạo usable run output; đã cleanup.

New session first probe:

```bash
true
```

Nếu probe pass, bỏ troubleshooting và chạy verification ngay. Nếu vẫn fail, locate actual generated/composed Bash wrapper, map physical line 192 và trace producer. Dùng exact standalone matching; không search substring `expo` vì sẽ match `export`/`module.exports`.

## Security incident

Một active-looking credential đã bị tool output làm lộ trong user-level Claude settings; trước đó một credential khác cũng bị lộ từ local settings. Không đọc lại, không trích dẫn, không serialize, không commit. Coi credentials đã compromise và rotate ngoài repository ngay.

Không đưa credential value vào prompt, report, bundle hoặc chat mới.

## Verification sequence sau khi Bash hoạt động

### 1. Capture preservation hashes

Trước build, hash:

- `notebooks/aic-kis-kaggle.ipynb`
- `dist/aic-retrieval-kaggle-contrastive-fix.zip`
- nếu cần, `config/reranker-disabled.example.yaml`
- nếu cần, `docs/evaluation-protocol.md`

Dùng lại hashes sau cùng để chứng minh không đổi trong session mới.

### 2. Focused tests

```bash
python -m unittest tests.test_reranking tests.test_retrieval tests.test_kaggle_bundle -v
```

### 3. Full local gates

```bash
python -m unittest discover -s tests -v
python -m compileall src scripts
git diff --check
```

Không coi `git diff --check` là bằng chứng đầy đủ cho untracked files.

### 4. Notebook AST và invariants

Parse mọi code cell của cả hai notebooks bằng:

```python
ast.parse(source, feature_version=(3, 10))
```

Verify:

- baseline: `ENABLE_RERANKING=False`
- Contrastive: `ENABLE_RERANKING=True`
- weights `0.1/0.1/0.0`
- budget `1.5`
- cap `96`
- hai warm-ups đứng trước reranker construction và first real query
- chỉ aggregate planner telemetry được serialized
- không output raw query hoặc generated plan
- privacy assertions và 100-unique-response checks còn nguyên

### 5. Deterministic double-build

Builder:

`scripts/prepare_kaggle_bundle.py`

Build hai temporary ZIPs, yêu cầu SHA-256 giống hệt nhau. Inspect members:

- sorted paths
- relative safe paths
- source-only
- không nested ZIP
- không `.avi`, `.faiss`, `.mkv`, `.mp4`, `.npy`, `.npz`, `.pdf`, `.webm`
- không path components `.git`, `plans`, `data`, `dataset`

Builder contract:

- timestamp `(2026, 1, 1, 0, 0, 0)`
- DEFLATE level 9
- fixed Unix regular-file mode

### 6. Final artifact và checksum pin

Build:

```bash
python scripts/prepare_kaggle_bundle.py --output dist/aic-retrieval-kaggle-planner-latency.zip
```

Compute SHA-256. Replace notebook placeholder with exact final SHA-256. Notebook không nằm trong ZIP, nên checksum pin không tạo recursion.

Sau pin, rerun:

- Contrastive notebook Python 3.10 AST parse
- notebook invariant checks
- bundle checks/tests
- final artifact SHA verification
- preservation hashes

Chỉ mark task `#15` complete khi mọi local gate pass và artifact/checksum khớp.

## Kaggle authoritative acceptance

Local completion không chứng minh T4 latency. Fresh Kaggle NVIDIA T4 run phải dùng **Save Version → Save & Run All** và thỏa:

- hai warm-ups thành công
- mỗi generated-token count thỏa `0 < count < 96`
- query thật parse thành plan
- real generated-token count thỏa `0 < count < 96`
- `planner_elapsed_ms < 1500`
- `applied=true`
- `fallback=false`
- `circuit_open=false`
- `experiment_valid=true`
- source bundle SHA khớp artifact mới
- scoring/privacy assertions pass

Nếu valid plans liên tục hit `96` hoặc real planner vẫn vượt `1.5 s`, giữ fallback và reject planner/runtime này. Không tăng cap hoặc budget.

## Paste prompt cho session mới

```text
Tiếp tục task AIC 2026 từ handoff sau, không hỏi lại quyết định đã rõ và không làm lại task hoàn tất.

Project: E:\15.DEEPLEARNING\Aichallenge2
Handoff: E:\15.DEEPLEARNING\Aichallenge2\plans\reports\handoff-260808-0830-planner-latency-bundle-report.md
Approved plan: C:\Users\minht\.claude\plans\splendid-fluttering-swing.md

Đọc handoff, plan và TaskList trước. Task duy nhất còn mở là #15 “Build và xác minh bundle”; #12-14 đã complete. Giữ #15 in_progress tới khi có fresh local verification.

Đầu tiên chạy Bash probe `true`. Nếu pass, bỏ điều tra executor và thực hiện đúng verification/build/checksum-pin sequence trong handoff. Nếu vẫn fail với `/usr/bin/bash: line 192: expo: command not found`, chẩn đoán generated/composed startup wrapper theo read-only-first: locate physical/generated line 192 và producer của standalone `expo`; không sửa settings/hooks/profile/snapshot khi chưa có bằng chứng. Dùng exact token matching, không substring `expo` vì false-positive `export`.

Ràng buộc bắt buộc:
- Không đọc `C:\Users\minht\.claude\settings.local.json`.
- Không đọc hoặc in credential values từ bất kỳ settings file nào. Credentials đã bị lộ phải được coi là compromised và rotate ngoài repo.
- Không commit, push, upload, mutate PDF, đọc private Kaggle labels, hoặc ghi đè artifact cũ.
- Giữ `planner_budget_seconds=1.5`, `MAX_NEW_TOKENS=96`, strict `0 < generated_tokens < 96`, exact fallback và circuit breaker.
- Không sửa `notebooks/aic-kis-kaggle.ipynb`, `config/reranker-disabled.example.yaml`, `docs/evaluation-protocol.md`, hoặc `dist/aic-retrieval-kaggle-contrastive-fix.zip`.
- Tạo final artifact `dist/aic-retrieval-kaggle-planner-latency.zip`, deterministic double-build, inspect ZIP, pin SHA-256 vào Contrastive notebook, rồi rerun post-pin checks.
- Không claim pass/completion nếu chưa có fresh command output.
- Promotion vẫn blocked pending private p50/p95 và labeled R@k/Final Score.

Báo chính xác command nào pass/fail, SHA-256 final, preservation evidence và phần Kaggle T4 còn phải chạy.
```

## Unresolved

- Actual generated Bash wrapper/source chứa standalone `expo` line 192 chưa được locate.
- Local tests/build/AST sau notebook edit cuối chưa chạy.
- Final ZIP chưa tồn tại; checksum placeholder chưa được thay.
- Fresh Kaggle T4 acceptance chưa chạy.
- Exposed credentials cần rotate ngoài repository.
