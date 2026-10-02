# Streaming and chunked prefill validation — 2026-10-03

## Implementation and correctness

The engine streams bounded token events while its actor continues to own every
session and KV page. Terminal replies have their own mailbox. A full event
buffer terminates that stream; disconnects, write failures, and shutdown cancel
work and release reservations. UTF-8 fragments are retained until complete, with
invalid-byte replacement matching non-streaming generation.

`PrefillBatch` flattens variable-length chunks across sessions, uses each row's
own causal prefix and RoPE position, and projects vocabulary logits only for
each session's last row. It validates cache ownership, reservations, context and
workspace bounds before allocating pages. Cache positions commit after the
final cancellation check. Cancellation is checked between layers; a running
matrix or attention kernel is not preempted.

A tick always advances every active session. The default chunk limit is 16 for
prefill-only ticks and 1 while any session is generating. The total token budget
is also enforced. The limits are configurable because larger chunks trade output
cadence for prompt throughput.

Validation commands passed:

```sh
go test -race -count=1 ./...
go vet ./...
CGO_ENABLED=0 go build -trimpath -o monolith .
./scripts/smoke.sh
```

Added tests cover exact sequential/chunked logits, full-forward parity, mixed
positions and KV page boundaries, cancellation and retry, reservation/workspace
limits, token budgets, concurrent submission during shutdown, stream/token/text
parity, fragmented UTF-8, slow consumers, real HTTP disconnects, asynchronous
errors, and network write-deadline cleanup. Write deadlines cover writes and
flushes, then clear during idle prefill and queue waits, including HTTP/2's timer
semantics. The CLI smoke test still produces byte-identical resumed checkpoints.

## Matched HTTP experiment

Environment: Apple M4, macOS 26.4.1, Go 1.27.1 darwin/arm64, four Go workers.
Server and load client ran on the same machine over loopback HTTP/1.1. Other host
activity was not controlled. This is a short local experiment, not a capacity
limit, long-running memory-soak test, or cross-language comparison.

The same 4,296,448-parameter `small` checkpoint was used throughout. It was trained
for only one update to create a reproducible artifact; its output is useful for
exercising the runtime and carries no language-quality claim.

Each trial starts a fresh server, warms it with four requests, then measures 16
requests with concurrency 4. Eight prompts contain 4 bytes and eight contain 180
bytes (5 and 181 tokens including BOS). Each produces 24 token events. All 144
measured requests across nine trials completed with exact stream/final-response
parity. Server settings: active sessions 4, queue 32, KV budget 64 MiB, token
budget 64, event buffer 64. No cancellations or slow readers are mixed into the
timing comparison.

Three variants were run in rotating order across three trials:

- **Sequential**: prefill chunk 1, mixed prefill chunk 1.
- **Bulk**: prefill chunk 16, mixed prefill chunk 16.
- **Default policy**: prefill chunk 16, mixed prefill chunk 1.

Values below are medians of three per-trial results. p95 uses nearest rank;
there are only eight requests per prompt class per trial, so each trial's TTFT
p95 is its largest observed TTFT. These are descriptive measurements, not
confidence bounds or service-level guarantees. Token gaps are measured at the
client and can include HTTP buffering. RSS was sampled externally with `ps` at
100 ms; Go heap and KV metrics were sampled at 50 ms. Sampled peaks can miss
short-lived maxima.

| Measurement | Sequential | Bulk | Default policy |
| --- | ---: | ---: | ---: |
| Short prompt TTFT p95 | 16.53 ms | 24.41 ms | 37.34 ms |
| Long prompt TTFT p95 | 487.93 ms | 157.80 ms | 313.08 ms |
| Short prompt token gap p95 | 2.53 ms | 13.94 ms | 2.64 ms |
| Long prompt token gap p95 | 3.63 ms | 11.58 ms | 3.95 ms |
| Short prompt completion p95 | 68.36 ms | 253.80 ms | 90.47 ms |
| Long prompt completion p95 | 546.21 ms | 365.51 ms | 386.69 ms |
| Completed output token events/s | 303.14 | 368.77 | 375.97 |
| Sampled peak RSS | 68.91 MiB | 68.52 MiB | 67.00 MiB |
| Sampled peak Go heap allocation | 32.93 MiB | 27.64 MiB | 29.85 MiB |
| GC cycles during measured requests | 1 | 0 | 1 |

In this workload, the default policy delivered **1.24×** as many completed
output token events per second and reduced long-prompt TTFT p95 by approximately
**36%** relative to sequential prefill. Short-prompt TTFT and total
completion time increased. Bulk prefill made long prompts faster again, but
substantially widened token gaps during ongoing generation. The default keeps
that tradeoff visible and favors output cadence. These findings are specific to
this model, workload, implementation and machine.

All comparison trials ended with zero active requests, queued requests and KV
reservations. Allocated KV pages remain cached for reuse. The KV limit does not
cover model weights, runtime heaps, token/event buffers, or total process RSS.

## Cancellation, slow readers and overload

Two separate loads used the default policy and an 8 MiB KV budget:

- Mixed readers: 36 requests, concurrency 8, queue 32; every fifth request
  canceled after two token events and every third reader delayed 10 ms between
  token events. Outcomes: **29 completed, 7 canceled**, no overload or
  protocol/transport errors. The server also recorded seven cancellations.
- Overload: 64 requests, concurrency 32, queue 4, with the same reader/cancel
  selection. Outcomes: **5 completed, 1 canceled, 58 overloaded**. Rejected
  requests were not retried.

Both loads drained to zero active requests, queued requests and KV reservations.
Their server totals include four warmup requests. Short responses fit in socket
buffers: these runs recorded no slow-consumer termination. The deterministic
unconsumed-stream test independently verifies that a full event channel releases
its reservation without blocking healthy sessions. Neither observation proves a
long-running RSS plateau.

## Reproduction and receipts

[Machine-readable receipts](benchmarks/serving-2026-10-03.json) contain per-trial
summaries, metrics snapshots, sampled RSS, model configuration and SHA-256
fingerprints. Individual generated text and local machine paths are omitted.
The JSON's `runs/` fingerprints identify local generated artifacts; those
binaries and checkpoints are not distributed.

```sh
mkdir -p runs/serving
CGO_ENABLED=0 go build -trimpath -o runs/serving/monolith .
go build -trimpath -o runs/serving/loadtest ./tools/loadtest
runs/serving/monolith train -data examples/tiny.txt -preset small \
  -steps 1 -warmup 0 -batch 1 -seq 8 -threads 4 \
  -save-every 0 -eval-every 0 -out runs/serving/small.mglm

runs/serving/monolith serve -model runs/serving/small.mglm \
  -addr 127.0.0.1:8080 -threads 4 -max-batch 4 -queue 32 -kv-mib 64 \
  -prefill-chunk 16 -mixed-prefill-chunk 1 -token-budget 64 -stream-buffer 64
```

In another terminal, prepare the same long prompt, warm up with `-requests 4`,
then measure with `-requests 16`:

```sh
long_prompt=''
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  long_prompt="${long_prompt}the cat rests. "
done
runs/serving/loadtest -url http://127.0.0.1:8080 \
  -requests 16 -concurrency 4 -max-tokens 24 \
  -short-prompt 'the ' -long-prompt "$long_prompt" \
  -timeout 60s -sample-interval 50ms -details > runs/serving/result.json
```

Restart the server between trials. Rotate the sequential, bulk and default
settings as recorded in the receipts. For separate adversity runs, use the
reader/cancellation flags documented in [the load client](tools/loadtest/README.md).
The client reports Go heap usage; collect process RSS separately if needed.

Earlier kernel-only tests also measured chunked prefill with `demo` and `tiny`:

```sh
go test -run '^$' -bench BenchmarkPrefillSequentialVsChunk \
  -benchmem -benchtime=5x -count=5 -cpu=4
```

Those warm-cache kernel measurements exclude HTTP, queueing, sampling and delivery
and should not be substituted for the end-to-end comparison above.
