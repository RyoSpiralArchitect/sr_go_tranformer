# Reusable inference workers

Each inference engine owns a fixed set of Go compute workers for matrix
projections and attention. The actor sends disjoint output ranges through a
bounded channel and waits for every range before starting the next kernel.
Workers are created once and joined when the engine closes.

```sh
./monolith serve -model runs/model.mglm -threads 4 -kernel-workers 4
```

`-kernel-workers` accepts 0–256. The default 0 captures
`min(runtime.GOMAXPROCS(0),256)` when the engine is constructed. Explicit values
2–256 create exactly that many worker goroutines; 1 executes on the actor with
no background compute workers. Small kernels still execute inline using the
existing work threshold. Runtime changes to `GOMAXPROCS` do not resize an
existing pool. `-threads` remains the process-wide Go execution setting.

## Ownership and bounds

- Each engine has its own executor and inference workspace. Engines can share
  an immutable model, but never share their job channels, scratch arrays,
  KV pages or worker shutdown.
- A kernel submits at most `min(workers, output ranges)` jobs. Both the job
  mailbox and completion mailbox have capacity `workers`, independent of the
  request queue, context length and model size. There is only one kernel in
  flight per engine, and no nested dispatch.
- Workers clear their borrowed job closure before acknowledging completion.
  Idle workers retain no job reference to a workspace or KV pages.
- Every kernel joins all its writes before the actor proceeds. Workspace return,
  cache-position commits, KV release and `CompletionStream.Released()` retain
  the [serving lifecycle contract](SERVING_LIFECYCLE.md).
- `Engine.Close()` drains requests and joins the executor before returning,
  including an engine closed before its first request. The number of engines
  multiplies the total worker count; this is a per-engine bound.

`monolith_kernel_workers` and `EngineStats.KernelWorkers` report owned compute
worker goroutines, including idle workers. The gauge becomes nonzero when the
actor starts the pool and returns to zero after worker teardown. It remains
nonzero when a healthy open engine has drained all requests. The separate
`monolith_computing_sessions` gauge counts sessions in a shared batch, not busy
workers. Runtime/HTTP goroutines are also outside the worker gauge.

The actor retains its workspace across requests and garbage collections. It
consumes borrowed logits before the next batch; responses and stream events
contain their own token/text results. Public `Model.PrefillBatch` and
`DecodeBatch` still return independent logit slices and use `sync.Pool` for
scratch reuse.

Workspace float capacities plus one reserved public logit copy stay within
the existing 512 MiB activation budget. Mixed shapes discard old capacities
when their combined retention would exceed that bound. Metadata is separately
bounded by 1,024 rows and 256 sessions: fixed row/position/index arrays and
reused validation maps. Cache and pool references are cleared after every
completed or failed computation, once all kernel borrowers have joined.

`monolith_workspace_float_bytes` and `EngineStats.WorkspaceFloatBytes` report
retained float array capacity after each tick, excluding metadata, weights,
KV storage, allocator overhead and garbage awaiting collection. The gauge
stays nonzero while an idle engine retains reusable scratch, and returns to
zero during shutdown after workers join. This is a per-engine retention
measurement, not process RSS or a promise that the runtime immediately returns
memory to the OS. See [measurements](INFERENCE_PERFORMANCE.md).

## Arithmetic and cancellation

Workers partition independent outputs and retain the original reduction order
within every output. Four-row projection tiles, attention reductions, FP32
storage and random sampling are unchanged. Training and evaluation keep their existing paths. Standalone
`Model.PrefillBatch` uses the default dispatcher with the same inference
head/lane partition; the engine passes its fixed executor explicitly, so a
shared model does not acquire mutable executor state. See
[long generation](LONG_GENERATION.md) for head partitioning and score scratch.

Request cancellation is checked between scheduling ticks. A shared batch uses
the engine context, so canceling one request does not abort healthy peers.
Engine cancellation is still checked between layers and before committing cache
positions. Worker reuse does not make a running matrix or attention kernel
interruptible. A successful join is required even during shutdown.

## Validation

```sh
go test -race ./...
go test -race -run 'Test(KernelExecutor|Worker|EngineWorker|ReleaseWaitsForMixedBatchCompute|ShutdownReleaseWaitsForComputeAndDrainsQueue)' -count=10 .
go vet ./...
./scripts/smoke.sh
```

Tests compare logits and KV float bits with the original dispatcher for worker
counts 1, 2 and 4, across MQA/GQA/MHA, mixed chunks and decode, page crossings,
and row/output tails. Single-row and tiled projections also match an independent
scalar reference bit for bit. Canceled batches preserve committed prefixes and can retry
with different input. Gated kernels prove dispatch cannot return while workers
still borrow its outputs. Mixed-request gates cover cancellation and resource
release using projections large enough to execute on the workers. Two engines
serve the same immutable model concurrently and can shut down independently.
The CLI smoke checks exact training resume and evaluation equality.

## Measurement

```sh
go test -run '^$' -bench '^BenchmarkInference(Linear|Batch)Workers$' \
  -benchmem -benchtime=300ms -count=3 -cpu=4
```

`spawn` selects the original dispatcher and `reuse` selects fixed workers in
the same test binary. The linear fixture projects 256 inputs to 704 outputs.
The batch fixture uses the `small` preset, four sessions, 96 cached prefix tokens
per session, and either one or 16 new tokens per session. Model, page allocation,
workspace warmup and worker startup are outside the timed interval. Returned
logits and normal batch bookkeeping are included. A forced GC may still evict
`sync.Pool` scratch, as in production.

The source-bound [measurement receipt](benchmarks/inference-workers-2026-10-15.json)
records all samples and the matched HTTP comparison. These are CPU runtime
measurements of a small local fixture, not a model-quality or general capacity
claim. Execution tracing is disabled during timing comparisons.

On Apple M4 / Go 1.27.1 at four execution threads, medians of three kernel runs
were:

| Kernel | Spawn allocs/op | Reuse allocs/op | Spawn B/op | Reuse B/op | Spawn / reuse time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projection, 1 row | 6 | 1 | 304 | 96 | 60.40 / 60.83 us |
| Projection, 4 rows | 6 | 1 | 304 | 96 | 70.95 / 68.88 us |
| Projection, 64 rows | 6 | 1 | 304 | 96 | 758.43 / 674.79 us |
| Batch, 1 token/session | 209 | 54 | 16,595 | 10,128 | 2.748 / 2.707 ms |
| Batch, 16 tokens/session | 299 | 54 | 21,280 | 11,089 | 25.097 / 24.849 ms |

Allocation reduction is consistent; fixed-batch timing ranges overlap. The one
remaining projection allocation is its closure, while a warmed executor dispatch
with an already-created callback allocates zero objects.

The HTTP comparison used merged commit `b5d428c` as baseline and this source as
candidate, alternating `baseline, candidate, candidate, baseline, baseline,
candidate`. Each process used the same one-update `small` checkpoint and server
settings, four serial output checks, eight concurrent warmup requests, then 72
mixed requests at concurrency eight. Every run completed 58 streams, canceled
14 requests as planned, passed stream validation and drained all request resources.
The four serial completion responses matched across all processes.

| HTTP statistic (median of 3 run statistics) | Baseline | Fixed workers |
| --- | ---: | ---: |
| Completed token events/s | 314.94 | 334.74 |
| Short, normal-reader first token p95 | 313.39 ms | 270.57 ms |
| Long, normal-reader first token p95 | 641.92 ms | 570.51 ms |
| Short, normal-reader token gap p95 | 2.698 ms | 2.713 ms |
| Long, normal-reader token gap p95 | 4.220 ms | 3.397 ms |

Throughput rose 6.29% in this fixture, while short-reader p95 token gaps slightly
increased. Candidate short-reader first-token p95 varied from 258.57 to 342.75 ms.
Each prompt group has only 19 healthy normal-reader completions per run, making
its nearest-rank first-token p95 the maximum observation. These are medians of
run quantiles, not a pooled quantile or a confidence interval. No uniform latency
improvement is established. Sampled process goroutine peaks were 27 for both
versions; reuse reduces churn, not necessarily the instantaneous goroutine count.
No GC occurred during these measured HTTP cohorts, so they do not establish a
reduction in GC frequency.

To repeat the HTTP workload, build each revision with the same Go toolchain and
`CGO_ENABLED=0`, prepare the [one-step fixture](SERVING_VALIDATION.md#reproduction-and-receipts)
once, and start each binary with the server flags recorded in the receipt. Use a
fresh process for each run. With the 180-byte `long_prompt` from that recipe:

```sh
go run ./tools/loadtest -requests 8 -concurrency 8 -max-tokens 24 \
  -short-prompt 'the ' -long-prompt "$long_prompt" \
  -cancel-every 0 -slow-every 0 -sample-interval 50ms > warmup.json
go run ./tools/loadtest -requests 72 -concurrency 8 -max-tokens 24 \
  -short-prompt 'the ' -long-prompt "$long_prompt" \
  -cancel-every 5 -cancel-after 2 -slow-every 3 -slow-delay 10ms \
  -timeout 30s -sample-interval 50ms -drain-timeout 5s -details > measured.json
```

The receipt retains all six run summaries, the complete kernel samples and raw
artifact fingerprints, including regressions. Binaries, the checkpoint and
individual request details remain local generated artifacts.
