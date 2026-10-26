# Inference performance

Measure token selection separately from model projection, attention, source
loading and HTTP delivery. A faster sampler does not imply the same speedup
for end-to-end generation. The sampler/workspace measurements use FP32 weights;
the last section covers exact-output Q4 kernel optimization. Separate
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

## Sharing Q4 weights across short batches

When a Q4 tile has two or three rows left, the first two now share each packed
weight read and pair-table lookup. A final odd row still uses the single-row
path; full four-row tiles retain their existing path. This applies to Q4
projections inside `q4-mixed` as well as all-Q4 models. Q8 and FP32 arithmetic
are unchanged.

Each row keeps exactly the same four accumulators, reduction order and block
scale multiplication as before. The per-row wide fallback, disjoint worker
output ranges and joined workspace ownership are unchanged. There is no new
activation quantization, expanded weight matrix, workspace or model allocation.
This reduces repeated decoding of weights; it does not reduce resident weight
payload or the precision of weights/activations/KV.

The benchmark retains the old kernels from `167adfb4` in test code and runs both
implementations in the same binary. On Apple M4 / 24 GiB / Go 1.27.1, four Go
threads/workers, three 300 ms samples in each run order, the 18,944 × 3,584
projection used by Qwen2.5-7B produced these medians:

| Input rows | Before, ms | Shared pair, ms | Time reduction |
| ---: | ---: | ---: | ---: |
| 1 | 6.577 | 6.711 | -2.0% |
| 2 | 13.274 | 8.890 | 33.0% |
| 3 | 19.709 | 15.614 | 20.8% |
| 4 | 13.411 | 13.416 | 0.0% |
| 6 | 28.161 | 26.801 | 4.8% |

The 8,960 × 1,536 case reduced two-row time by 30.1% and three-row time by
19.8%. One/four-row controls remain close; no speedup is claimed for them.
The kernel retains 112 B and one callback allocation per call. Desktop/thermal
activity was uncontrolled, so these samples are workload/host-specific.
[Raw samples](validation/performance/q4-pair-kernels-20261026.txt) and the
[aggregate/build manifest](validation/performance/q4-pair-kernels-20261026.json)
retain every trial, including regressions.

```sh
GOMAXPROCS=4 ./scripts/go.sh test -run '^$' -bench '^BenchmarkQ4Pair$' \
  -benchmem -benchtime=300ms -count=3 -cpu=4
```

Exact-bit regression tests cover zero/odd/even row counts, 32-output tile
boundaries, odd matrix widths, partial 64-weight blocks, subnormal/signed-zero
inputs, very large activations and nil/one/four-worker execution. Existing HF
profile, workspace, cancellation and shared-model lifecycle tests also apply.

### Bounded real-model comparison

The offline driver builds the selected Git baseline and current `monolith.go`
in separate directories using the identical `inference_perf_test.go` harness.
It requires pinned original model files from the catalog and an already
installed Go toolchain/dependency cache. It creates no model download or
expanded FP32 weights. The driver disables the user Go environment file,
ambient build flags/workspaces, automatic toolchain downloads and module
downloads, and records the effective build settings.

```sh
python3 -S scripts/compare_inference.py --model PATH_TO_QWEN_7B \
  --id qwen25-7b --baseline-ref 167adfb4 \
  --output-dir runs/qwen7b-pair-comparison
```

Only one mixed-format model is resident at a time. The process order is
before/after/after/before. Each process measures one- and two-session cases,
with one warmup and three measured trials per case. Each session starts with
eight tokens from the fixed authored corpus, then performs seven decode calls
with its own greedy IDs. This fixed-step probe continues even if an EOS ID
appears; request termination is covered by the serving tests.

Timers surround only the model's forward calls. Loading, tokenizer work,
hashing, token selection and HTTP/SSE are excluded, so these are core-forward
latencies, not end-to-end request throughput. Allocation totals include the
measurement loop. SHA-256 digests of every produced logit and all used KV
values, exact generated IDs, prompt IDs and workspace capacities must agree
between variants. Per-child peak RSS includes the entire process lifetime and
is separate from both allocated weight payload and retained workspace bytes.

### Qwen2.5-7B result

The pinned `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28` was tested offline on the same
Apple M4 / 24 GiB / Go 1.27.1 host. Source files were SHA-256 checked before
and after the run. All four processes completed; both versions produced
identical full-logit and used-KV digests, generated IDs and workspace capacities.

| Sessions | Before: seven decode calls, s | Shared pair: seven decode calls, s |
| ---: | ---: | ---: |
| 1 | 4.930 (4.518–6.717) | 5.415 (4.471–7.020) |
| 2 | 10.174 (9.271–14.046) | 6.675 (6.636–7.791) |

Values are medians and full ranges of six measured trials per version/case.
Two-session decode time fell 34.4%. The single-session median was 9.8% slower
and its ranges overlap; this run does not establish a single-session speedup.
Prefill controls also varied, and no prefill improvement is attributed to the
two-row change. The two-session decode intervals contain fourteen token
positions across seven calls; these are not eight-token request durations or
HTTP throughput measurements.

Both versions keep 4,829,927,424 weight bytes (4.498 GiB), 3,033,344 workspace
bytes for one session and 6,066,688 bytes for two. The two-session measurement
loop's median allocated bytes stayed 7,589,120, with 1,830 allocations; this
includes KV pages, probe hashing and token bookkeeping, not just kernels.
Per-process peak RSS was 4.114/4.086 GiB before and 4.036/4.095 GiB after.
No RSS reduction or minimum-RAM claim is made; OS residency and allocated
weight payload are different quantities, as in the
[earlier 7B quality comparison](QUALITY_INFERENCE.md#measured-7b-results).

The [complete comparison receipt](validation/performance/qwen7b-q4-pair-20261026.json)
retains all times, numerical signatures, resource counts and build/model hashes.
It was produced before the runner's Go-environment isolation fix. The host's
effective GOFLAGS/GOEXPERIMENT were empty; rebuilding both binaries under the
final isolated settings produced **identical binary and input hashes** to those
measured. The [isolation verification receipt](validation/performance/q4-pair-build-isolation-20261026.json)
records those builds, effective settings and a successful SmolLM2-360M pilot.
The original 7B receipt keeps its actual runner hash rather than replacing it
with the later script's identity.
