# Inference performance

Measure token selection separately from model projection, attention, source
loading and HTTP delivery. A faster sampler does not imply the same speedup
for end-to-end generation. The measurements below use FP32 weights. Separate
[Q8 measurements](Q8_INFERENCE.md) cover opt-in quantization and larger models.

## Greedy token selection

Temperature zero uses a single vocabulary scan instead of allocating and
sorting candidates. Ties still choose the lowest token ID. Repetition penalties
use the same float64 arithmetic; all logits, including blocked tokens, must be
finite. This path does not advance the request's RNG. Sampling at nonzero
temperature retains the existing sorter, top-k, top-p and RNG behavior.

With repeat penalty 1, greedy selection allocates no memory. Other penalties
still build a history membership map, whose size follows the request history;
they no longer allocate vocabulary-sized candidate arrays.

Measured on 2026-10-24, Apple M4, darwin/arm64, Go 1.27.1, four Go threads,
three trials of 300 ms. Values are medians. Both algorithms run in the same
benchmark with identical synthetic logits, a 256-token history and no blocked
token, as used by supported HF profiles. The retained sorted reference is the
implementation before this change (parent commit `40a31d6`).

| Vocabulary | Repeat penalty | Sorted reference | Single scan | Bytes/op before → after |
| ---: | ---: | ---: | ---: | ---: |
| 49,152 | 1 | 5.995 ms | 0.166 ms | 2,672,013 → 0 |
| 50,257 | 1 | 6.152 ms | 0.173 ms | 2,737,555 → 0 |
| 151,936 | 1 | 20.471 ms | 0.512 ms | 8,226,184 → 0 |
| 50,257 | 1.1 | 6.238 ms | 0.279 ms | 2,737,551 → 9,512 |

Allocation counts fall from 8 to 0 with penalty 1, or to 3 with penalty 1.1.
These are sampler microbenchmarks, not tokens/second or process RSS results.
[Raw trials](validation/performance/greedy-20261024.txt) include the native
258-token vocabulary and all measured penalty settings.

Reproduce:

```sh
GOMAXPROCS=4 ./scripts/go.sh test -run '^$' \
  -bench '^BenchmarkGreedySampling$' -benchmem -benchtime=300ms -count=3 -cpu=4
```

Compatibility tests compare against the retained sorter across ties, signed
zero, subnormal/extreme values, blocked IDs, repeated histories and invalid
distributions. They also compare nonzero-temperature choices and RNG states.
Pinned cached GPT-2 and SmolLM2-135M CLI generations matched the existing
validation JSON exactly with `HF_HUB_OFFLINE=1`.

## Engine-owned workspace

An engine keeps its float buffers and bounded row/validation metadata for its
lifetime. Its internal consumer borrows logits until the next tick, removing
one batch × vocabulary copy per tick. Public model calls still return owned
results. Reference clearing, shutdown and capacity bounds are described in
[inference workers](INFERENCE_WORKERS.md).

Two measurements isolate different effects. Both use the same host/toolchain
as above, four threads, warm caches and three trials:

| Measurement | Before | After |
| --- | ---: | ---: |
| Entire native engine request, allocated bytes | 40,606 B | 28,360 B |
| Entire native engine request, allocations | 515 | 475 |
| Entire native engine request, median duration | 0.837 ms | 0.833 ms |
| 4-session / 50,257-vocabulary decode batch, allocated bytes | 812,837 B | 1,800 B |
| Same decode batch, median duration | 0.620 ms | 0.626 ms |

The whole-request comparison runs `BenchmarkEngineWorkspaceRequest` unchanged
at parent commit `c248c99` and with this change: a synthetic native model with
dimension 64, hidden size 100, 2 layers, 4 heads / 2 KV heads, context 128,
258 vocabulary entries, a 33-token prompt and 8 greedy output tokens.
Both versions already have the greedy scan. The measured allocation reduction
is about 30%; the duration ranges overlap, so this does not establish a speedup.
Raw trials: [before](validation/performance/engine-before-20261024.txt),
[after](validation/performance/engine-after-20261024.txt).

The batch comparison runs both the public pooled/copying path and the internal
owned/borrowing path in the new implementation, with four sessions, 96-token
prefixes and one new row per session. It uses the same synthetic geometry but
50,257 vocabulary entries. Thus it isolates the result copy and pool access,
not all changes relative to the parent. Other vocabulary/chunk shapes are in
the [raw batch trials](validation/performance/workspace-20261024.txt).

```sh
GOMAXPROCS=4 ./scripts/go.sh test -run '^$' \
  -bench '^BenchmarkEngineWorkspaceRequest$' -benchmem -benchtime=500ms -count=3 -cpu=4
GOMAXPROCS=4 ./scripts/go.sh test -run '^$' \
  -bench '^BenchmarkInferenceWorkspace$' -benchmem -benchtime=300ms -count=3 -cpu=4
```

The owned workspace survives GC, so idle engines retain scratch until close.
Reducing allocation churn does not reduce weight storage, and no process RSS
reduction is claimed. Kernel callbacks still allocate in these measurements;
the complete inference path is not allocation-free.

F32/F16/BF16 Llama, Qwen2 and GPT-2 fixtures compare both output paths against
Transformers references and compare their logits and KV contents bit for bit.
Tests cover mixed shapes, capacity trimming, immutable public results, forced
GC, failed validation, cancellation/retry, shared models and independent engine
shutdown. Pinned real GPT-2 and SmolLM2-135M also passed CLI, HTTP, concurrent
SSE and resource-release checks against Transformers in offline mode.
Receipts: [GPT-2](validation/performance/gpt2-workspace-reference.json),
[SmolLM2-135M](validation/performance/smol-workspace-reference.json).

## Combined GPT-2 serving check

The pinned `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` was also measured through the Go
HTTP/SSE load client. This compares the pre-optimization binary at `40a31d6`
against greedy scan plus owned workspace together, with no model downloads.

Each fresh server uses context 128, four threads and kernel workers, maximum
batch 2, queue 4, 32 MiB KV budget, prefill chunk 8, mixed prefill chunk 4 and
16 tokens per tick. After two warmup requests, eight measured requests each
generate eight greedy tokens at concurrency 2. Short and long prompts
alternate. Three trials per binary alternate their run order.

| Version | Completed output tokens/s, median | Trial range |
| --- | ---: | ---: |
| Before both optimizations | 16.07 | 15.96–16.10 |
| Greedy scan + owned workspace | 18.14 | 18.08–18.29 |

This workload improved about 13%. Every measured request completed, with
64 output tokens per trial, successful resource drain and clean server
shutdown. Source loading and warmup are outside the timed interval. This is
one CPU/model/workload, not a general throughput guarantee. It does not isolate
the two optimizations or establish reduced resident weight memory.

The [raw serving receipt](validation/performance/gpt2-serving-20261024.json)
records binary and source hashes, exact settings, latency distributions,
sampled Go heap/GC metrics and drain results. Sampling peaks are neither
continuous peaks nor process RSS.

Use `serve` with the settings above and run the following client once with
`-requests 2` for warmup, then with `-requests 8` for the measurement:

```sh
./scripts/go.sh run ./tools/loadtest -url http://127.0.0.1:8080 \
  -requests 8 -concurrency 2 -max-tokens 8 -timeout 60s -sample-interval 50ms \
  -short-prompt 'The capital of France is' \
  -long-prompt 'Once upon a time, a small cat walked through the garden. Once upon a time, a small cat walked through the garden. Once upon a time, a small cat walked through the garden. Once upon a time, a small cat walked through the garden. '
```
