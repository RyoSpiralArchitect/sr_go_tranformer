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
go test -race -run 'Test(KernelExecutor|Worker|EngineWorker|ReleaseWaitsForMixedBatchCompute|ShutdownReleaseWaitsForComputeAndDrainsQueue)' -count=10 ./cmd/monolith
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
go test ./cmd/monolith -run '^$' -bench '^BenchmarkInference(Linear|Batch)Workers$' \
  -benchmem -benchtime=300ms -count=3 -cpu=4
```

`spawn` selects the original dispatcher and `reuse` selects fixed workers in
the same test binary. The linear fixture projects 256 inputs to 704 outputs.
The batch fixture uses the `small` preset, four sessions, 96 cached prefix tokens
per session, and either one or 16 new tokens per session. Model, page allocation,
workspace warmup and worker startup are outside the timed interval. Returned
logits and normal batch bookkeeping are included. A forced GC may still evict
`sync.Pool` scratch, as in production.

The source-bound [measurement receipt](../benchmarks/inference-workers-2026-10-15.json)
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

## Attention invocation reuse (2026-11-06)

This branch prototypes a workspace-owned attention callback and a cleared
invocation descriptor. Inline attention bypasses dispatch. Warm owned calls
allocate zero objects rather than one; scalar reduction order and lane planning
are unchanged. Gated tests verify the descriptor is not cleared before all
workers join. Callback storage adds bounded metadata, not a cross-request KV
pool or speculative payload allocation.

The [experiment index](../validation/performance/attention-direct-20261106.json)
retains all kernel and full-generation paired samples on Mac and Linux. Linux
kernel timings are near neutral, but the small Mac parallel condition includes
unfavorable observations and large reference-control drift. Full-generation
completion JSON matches throughout, but this does not resolve that kernel
question. The initial prototype remained a draft pending stable Mac characterization;
it is not a uniform latency or serving-throughput improvement claim. Direct
validation-map and KV-directory changes are reviewed separately.

A [controlled Mac remeasurement](../validation/performance/attention-control-mac-20261106.json)
adds reference/reference and candidate/candidate pairs for the four-worker,
single-row, 128-prefix condition. Each fresh process interleaves 16 trials of
all three comparisons, with four calls per block and balanced label order.
Exact finite output bits are checked outside timing after every block. The
runtime is unchanged; source and executable hashes are recorded in each receipt.

This session was heavily loaded (1-minute load approximately 101-108 on ten
logical CPUs). Same-reference medians were 1.320-1.341, while candidate/reference
medians changed direction across processes. These controls invalidate a latency
judgment for this session; they do not prove the cause of the earlier differences.
Keep the draft pending repeated low-load sessions. A successful timing test
means numerical checks passed, not that a performance gate passed.

```sh
CGO_ENABLED=0 ./scripts/go.sh test -c -trimpath \
  -o runs/attention-controlled-mac.test ./cmd/monolith
MONOLITH_ATTENTION_CONTROL_OUTPUT=runs/attention-control-fresh-run.json \
  runs/attention-controlled-mac.test \
  -test.run '^TestAttentionInvocationControlledTiming$' -test.v -test.timeout=2m
```

Use a new output name for each process; existing receipts are never overwritten.
Do not stop unrelated workloads merely to obtain favorable measurements.

### Primary-candidate selection and callpath contrasts

The [selection receipt](../validation/performance/attention-candidate-selection-20261106.json)
keeps compact descriptor/callback reuse (`cached-helper`) as the primary candidate.
A fresh compact callback (`transient-helper`) still allocates one object per call.
A cached direct-loop variant (`cached-captured`) also reaches zero warm allocations,
but needs larger state without demonstrating a supported latency advantage.
A matched-callback experiment snapshots `heads` before the task loop; it does not
demonstrate a benefit either, so no production rewrite was made for either option.
These contrasts also differ in copying, initialization and dispatch details;
they are not a perfectly isolated factorial experiment.

The new 64-call blocks interleave 32 trials per comparison with balanced order.
Self-control medians still reach approximately +5%/-6% in the second phase, and
outliers remain. Initial low-load calibration no longer reproduced the earlier
33% same-reference bias, but lower host load alone is not a latency guarantee.
All timings, including unfavorable observations, remain in the receipts.

Full-generation comparisons now use main after PR #55 (`9dc7d7a`) rather than the
pre-bookkeeping baseline. Three processes, five conditions and 24 alternating
pairs per condition compare every completion JSON outside timing. A process
median includes 5.5% slower prefill; do not claim a universal improvement or no
regression. The current choice is justified as verified invocation-allocation
reduction, not by selecting favorable timing samples. Test-only variants cover
72 layer/shape combinations with exact finite bits, warm allocations and
post-join zero-state checks. The runtime source remains unchanged in this study.

```sh
CGO_ENABLED=0 ./scripts/go.sh test -c -trimpath \
  -o runs/attention-callpaths.test ./cmd/monolith
MONOLITH_ATTENTION_CALLPATH_OUTPUT=runs/attention-callpaths-fresh.json \
  runs/attention-callpaths.test \
  -test.run '^TestAttentionCallpathTiming$' -test.v -test.timeout=2m
python3 -S -m unittest discover -s scripts -p 'attention_selection_receipts_test.py'
```

### Integration decision

After reviewing the allocation, numerical and lifecycle evidence, the user
requested integration of the compact reuse candidate. Adoption is explicitly
allocation-focused and conditional on the updated CI matrix. Variable Mac
controls and the slower prefill observation remain recorded risks, not resolved
latency gates or evidence of a uniform speedup. Historical draft/readiness notes
above describe their measurement checkpoints, not a claim that those risks
vanished. No worker-count, threshold or arithmetic-order tuning was adopted.

### Post-integration exploration

At merged baseline `2f3a8a1`, three eight-iteration runs of
`BenchmarkOwnedGenerate/tokens128/reversefalse/after` report 313/312/312 logical
allocations and approximately 445 KB allocated per operation. These runs enable
allocation profiling at rate one, so their latency is not a speed baseline.
A `generateInput`-focused profile excludes baseline-generator/model setup stacks;
profile percentages still include warmup and are not per-operation shares.

`Tokenizer.Pieces` builds 256 one-byte slices for every byte-token table, and
`Tokenizer.Decode` rebuilds that table when completing output. The next bounded
hypothesis is one per-call 256-byte backing block with one-byte, capacity-limited
slices, not a global mutable tokenizer cache. Verify byte/BOS/EOS/merge identity,
independent calls and mutation/append isolation before adoption. Retaining one
piece could retain the full block; that bounded retention tradeoff must remain
explicit. Tiny-allocation packing also means profile object counts need not equal
benchmark logical allocation counts.

Workspace resize and actual KV payload pages still dominate allocated bytes.
Do not treat necessary request storage as avoidable, or introduce speculative
pages/cross-request pools without separate retention and lifecycle evidence.

### Byte-piece packing prototype

The [prototype receipt](../validation/performance/tokenizer-pieces-packing-20261106.json)
records one per-call 256-byte block and full slices of length/capacity one, with
no global cache. Base-table construction falls from 257 to 2 logical allocation
operations; mixed and maximum-length merge fixtures each remove the same 255
operations. Exact piece/decode, special-token, concurrent-call and mutation/append
isolation tests pass. The allocation regression failed before the production edit.

Allocated bytes stay effectively unchanged because Go already packs tiny
one-byte allocations. A single retained byte piece now keeps the whole 256-byte
block alive; this is a bounded retention tradeoff, not a memory-footprint win.
HF tokenizers and merge copying remain unchanged.

The 128-token fixture reports 57–60 allocations versus the earlier main's
312–313, but these are separate runs with different profiling settings; use the
standalone frozen-reference test for the exact 255-operation reduction. Three
fresh-process, complete-runtime paired comparisons preserve every completion;
condition medians range approximately -0.5% to +0.9%. Slower observations remain
in the receipt. No uniform generation speedup, byte/RSS reduction or GC-frequency
improvement is claimed. These are the pre-PR prototype observations; the
[PR-readiness assessment](../validation/performance/tokenizer-pr-readiness-20261106.json)
records the subsequent verification and publication decision.

### Scoped byte-only Decode fast path

The [Decode receipt](../validation/performance/byte-decode-fastpath-20261106.json)
compares against the preceding packed-table prototype, not against the original
257-object table. Only a tokenizer with no merges bypasses table construction.
It fills at most 256 local scratch bytes, counts remaining emitted bytes only
when spilling, and returns an owned string. Large invalid-ID inputs do not cause
input-length-based buffer reservation. Merged/BPE and HF decoding remain unchanged.

For 128 valid byte IDs, separate microbenchmark blocks observed approximately
0.7 microseconds / 8 allocations / 7,160 bytes before versus 0.08 microseconds /
1 allocation / 128 bytes after. This multiplier is specific to that microfixture.
The all-negative-ID 8192-entry case is slower despite eliminating allocations;
retain that unfavorable result rather than claim universal Decode acceleration.
A general exact-size strings.Builder reduced allocation but was slower on
byte-heavy BPE input, so it was not adopted. Large merge expansions benefiting
from that experiment are recorded without selecting only those results.

A same-runtime tokenizer wrapper isolates Decode for generation. Three fresh
processes compare byte and merged controls across five shapes and 24 balanced
pairs per shape, checking every completion outside timing. Generation medians
change direction; unchanged merged controls also vary. No established whole-model
latency improvement, uniform non-regression, tail-latency or real-model throughput
claim follows from the microbenchmark. The byte 128-output benchmark saves about
seven logical allocations and 7 KB versus the packed Decode reference, separate
from the preceding Pieces-only reduction. Output ownership, concurrency and
255/256/257-byte spill boundaries with sparse/invalid IDs are covered by tests.

The [whole-generation qualification](../validation/performance/tokenizer-full-generation-assessment-20261107.json)
adds three processes of 64 pairs per condition against the pre-PR runtime and
identical-runtime controls, plus repeated stage comparisons: 2,640 timed pairs
with matching completions. The 128-token full-runtime process medians change
from +0.096% to -0.332%; aggregate -0.023% is not an established latency gain.
Raw receipts and their tracked Python validation are retained for review.
Allocation limits are compiler-sensitive upper bounds, not portable Go language
guarantees; the integration CI checks them on Go 1.22/Linux and stable Go on
Linux/macOS. The invalid-ID-heavy slowdown remains accepted and documented; its
all-negative case exits after one scan, so a second scan is not its established
cause. No additional production tuning was adopted based on these timings.

### Post-tokenizer CPU exploration

After PR #57 merged, [CPU/block exploration](../validation/performance/post-tokenizer-cpu-exploration-20261107.json)
on `05410f9` isolates the next hypotheses without changing production code.
The one-thread 128-token fixture attributes about 90% of sampled CPU to
`linearFP32Range`. Four-thread Darwin profiles instead skew heavily toward native
wait/signal frames; process counters and a separate block profile were collected
as checks, not as proof that those percentages are removable wall-clock overhead.
Block delay includes idle workers, parallel waits and the benchmark harness.

The next bounded compute hypothesis is sharing an input value across two output
accumulators in the FP32 decode/remainder-row path while preserving each output's
ascending input-index accumulation order. Four-row prefill arithmetic, task
partitioning, workers and thresholds should remain unchanged for the first
experiment. Exact bits, odd output tails, remainder rows, serial/parallel use and
whole-generation parity are prerequisites to evaluating latency. Dispatch and
completion joins remain a secondary hypothesis requiring matched controls;
profile percentages alone do not select a scheduling rewrite.

### Paired-output FP32 prototype

The [paired-output selection receipt](../validation/performance/fp32-outputs-selection-20261107.json)
records the first compute experiment against merged `05410f9`. Only the
remainder-row loop changes: two output accumulators share each input load while
each output retains its original ascending-index reduction order. Odd output
tails use the scalar loop; the four-row path, workers and thresholds are unchanged.
Frozen-reference tests cover 630 small shape combinations, larger serial/parallel
calls, special FP32 bit patterns, zero warm allocations and post-join cleanup.

Three fresh-process matched kernel comparisons show large decode reductions,
but preserve roughly 1% slower four-row and some mixed-row parallel observations.
Complete-runtime alternating comparisons also preserve every completion: the
128-token synthetic FP32 fixture has candidate/reference ratios 0.686/0.679/0.677,
with elapsed process medians around 42.4–42.7 ms versus 28.9–29.0 ms. Fresh runs of
the prior identical-runtime control binary stay near 1.000–1.001 for that shape.
Long-prefill/one-token differences are much smaller. This is a substantial local
fixture result, not a universal prefill, other-platform/toolchain, real-model,
quantized, tail-latency or memory improvement claim.

A [real-model receipt](../validation/performance/fp32-outputs-realmodel-20261107.json)
extends the check to Llama-3.2-3B-Instruct (f32, greedy, threads=4, Apple M4):
alternating fresh-process runs measure 0.666x reference for 32-token medians,
0.636x for a 64-token pair and about 0.615x for 256-token decode, with
byte-identical completions. A q8 control stays near 1.00 because quantized
weights use dedicated kernels the candidate does not touch. These remain
single-model, single-host observations.

## FP32 invocation reuse (2026-11-06)

The historical worker table above predates FP32 invocation reuse. A live executor
now holds one FP32 descriptor and one callback allocated at construction, rather
than allocating a projection closure for every kernel. The descriptor borrows
input/output slices and immutable weights only until all worker completions have
joined, then is cleared. Inline kernels use a stack descriptor directly. The
shared mailbox layout, worker counts, thresholds, Q4/Q8 kernels and reduction
order are unchanged. The nil-executor fallback remains independent and does not
promise allocation-free dispatch.

A warmed owned FP32 projection now allocates zero objects per call. This is an
allocation-volume result, not a peak RSS, GC-frequency or serving-throughput
claim. Startup adds a callback allocation and bounded descriptor storage. The
[source-bound experiment index](../validation/performance/fp32-dispatch-pointer-20261106.json)
retains three paired timing processes and the rejected alternatives, including
an unfavorable [typed-job comparison](../validation/performance/fp32-dispatch-20261106.json).

The paired fixture alternates reference/candidate order within one process,
checks exact finite output bits, and covers nil dispatch, one worker, four
workers and five matrix shapes. All samples are retained. Condition medians
across three processes were approximately within -2.2% to +2.0%; one individual
process median was 5.7% slower. These results do not establish a uniform latency
improvement or real-model/HTTP performance. `AllocsPerRun` temporarily uses one
Go execution thread; paired timing uses four.

The separate-process ABBA comparison also retains slower prefill observations
(about 7–9%) and larger excursions in unchanged quantized controls. To reduce
cross-process clock/scheduler confounding, a second driver puts package-renamed
copies of the complete baseline and candidate runtimes into one executable.
Across three fresh processes, five generation cases and 24 alternating pairs per
case, all completion JSON matched. Condition medians across processes were
approximately within -1.5% to +1.3%; individual process medians include +3.0%.
This does not erase the unfavorable ABBA samples or establish an HTTP/real-model
speedup. The driver includes common result-interface boxing in timing and keeps
JSON equality checks outside timing.

```sh
./scripts/go.sh test -run 'TestWorkerLinearSteadyAllocation' -count=1 ./cmd/monolith
CGO_ENABLED=0 ./scripts/go.sh test -c -trimpath -o runs/fp32-paired.test ./cmd/monolith
GOMAXPROCS=4 MONOLITH_DISPATCH_COMPARE_OUTPUT=runs/fp32-paired.json \
  runs/fp32-paired.test -test.run '^TestFP32DispatchPairedTiming$' -test.v
python3 -S scripts/compare_fp32_dispatch.py --baseline-ref 6fd5f01 \
  --output runs/fp32-dispatch-comparison --benchtime 1s --count 3
```

Use a fresh output directory for the process-level comparison. It compiles the
same shared tests with each runtime in isolated flat modules and uses ABBA
process order. Its output retains all samples, source/binary fingerprints and
process resources. Synthetic completion checks are included; untimed fixture
setup and benchmark reference calls also contribute to process RSS. Do not
interpret benchmark `after` names as a second runtime revision: those names
select the existing production-path variant inside each binary. Cross-compiled
Linux logs can be imported with `scripts/assess_dispatch_logs.py`; the rejected
comparison records why source-level allocation savings alone are insufficient.

To build the same-process full-generation driver (use `--target linux-amd64`
only when cross-compiling for Linux):

```sh
python3 -S scripts/compare_generation_pair.py --baseline-ref 6fd5f01 \
  --output runs/generation-pair-comparison
runs/generation-pair-comparison/generation-pair \
  -output runs/generation-pair-comparison/run1.json -trials 24
```

Run it in three fresh processes with distinct receipt names. The driver copies
both complete runtimes into temporary comparison packages; this does not split
the production monolith. The frozen receipt tests verify file fingerprints and
recompute paired medians without pinning future runtime sources to old receipts.

## Direct workspace metadata and KV directories (2026-11-06)

Fresh single-session direct generation does not need duplicate-cache detection
or pooled-page reservation aggregation. Those maps are now initialized only for
multiple sessions or an actual page pool, respectively. Reused workspace maps
are still cleared after each call, including failures. Duplicate-cache,
reservation and budget checks are unchanged.

Bulk KV growth reserves the page-pointer directory once for the actual required
page count. Payload pages remain lazy and separately allocated; there is no
cross-request pool or speculative allocation for future generated tokens.
A fresh 32-page fixture drops from 70 to 65 allocation objects. Standard Go slice
growth can change directory capacity; this is not a promise about peak RSS.

The [source-bound receipt](../validation/performance/direct-kv-20261106.json)
retains three Linux paired-generation processes with exact completion-JSON
comparisons. Condition medians were approximately -1.7% to +0.7%; no uniform
latency, serving-throughput or model-quality improvement is established.
Attention callback reuse is deliberately separate from this bookkeeping change.
