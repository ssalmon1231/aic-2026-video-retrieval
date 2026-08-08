# Qwen planner/runtime rejection

## Decision

Reject current `Qwen/Qwen2.5-0.5B-Instruct` planner/runtime for AIC 2026 Contrastive Text path. Stop prompt-only tuning. Keep exact-baseline fallback and fail-closed circuit breaker authoritative.

`dist/aic-retrieval-kaggle-planner-latency.zip` SHA-256 `5adc5701a40a3468ac7c1c8dc63ca1df81480ce460336da42aedabb2362dc92c` remains reproducible failure evidence, not promotion candidate.

## Fresh Kaggle NVIDIA T4 evidence

| Source SHA-256 prefix | First warm-up | Generated tokens | Result | Private-safe diagnosis |
|---|---:|---:|---|---|
| `bdbf22a8aea2` | 2897.17 ms | 40 | failed | `planner text list exceeds its bound` |
| `870abd01be5d` | 2323.43 ms | 42 | failed | malformed JSON; object opened but never closed; odd quote count; `Expecting ':' delimiter` |
| `5adc5701a40a` | 1727.33 ms | 28 | failed | aggregate run confirms post-generation failure; exact subtype not serialized |

All counts satisfy `0 < count < 96`; failures are not cap truncation. Warm-up elapsed values are startup diagnostics, not real-query `1.5 s` budget measurements. Every run failed before second warm-up, reranker construction, and real planner call. Exact fallback stayed active; privacy summary stayed clean.

## Why constrained decoding is not added now

Transformers exposes low-level `prefix_allowed_tokens_fn` and custom `LogitsProcessor` hooks. They require project-owned token grammar/state logic; they do not provide a native JSON-schema guarantee. `force_words_ids` can force token phrases but cannot enforce object syntax, key order, delimiters, array bounds, or semantic validity.

A robust JSON solution therefore requires one of:

- custom tokenizer-aware grammar masking;
- custom generation loop;
- new constrained-generation dependency.

Each option adds implementation, compatibility, reproducibility, and per-token latency risk under an unverified `1.5 s` T4 gate. Syntax constraints also cannot prove useful positive/negative semantics from the 0.5B model. Current evidence does not justify that scope.

References:

- <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation>
- <https://huggingface.co/docs/transformers/main/en/internal/generation_utils>
- <https://huggingface.co/docs/transformers/main/en/generation_strategies>

## Preserved contracts

- `MAX_NEW_TOKENS=96`.
- `planner_budget_seconds=1.5`.
- Strict four-key parser and maximum three negatives.
- `do_sample=False`, pinned model/revision, FP16 CUDA.
- Exact-baseline fallback and in-memory fail-closed circuit breaker.
- No raw query, prompt, generated plan, vectors, labels, credentials, or frames serialized.
- Baseline notebook, old artifact, disabled config, and evaluation protocol unchanged.

## Future constrained-decoding experiment

Only reopen as separate approved scope. Minimum acceptance:

1. No remote executable code and no unpinned dependency.
2. Strict parser remains final verifier.
3. Generated content stays memory-only.
4. Two representative warm-ups and real query all parse with `0 < generated_tokens < 96`.
5. Post-warm-up real planner latency `< 1500 ms` on NVIDIA T4.
6. `applied=true`, `fallback=false`, `circuit_open=false`, `experiment_valid=true`.
7. Locked private multi-query p50/p95 plus labeled R@k/Final Score justify added complexity.

## Status

Current planner candidate rejected. Promotion blocked. Exact baseline remains valid path.

## Unresolved questions

- Whether to authorize a separately scoped constrained-decoding experiment.
- Private post-warm-up p50/p95 and labeled R@k/Final Score remain unavailable.
