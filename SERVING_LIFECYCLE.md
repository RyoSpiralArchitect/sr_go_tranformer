# Serving lifecycle measurement

The inference engine owns each accepted request from enqueue until resource
return. A request can stop waiting before the engine has finished using its KV
pages. This distinction matters when a cancellation arrives during a mixed batch.
The current scheduler and arithmetic are unchanged by this instrumentation.

## Completion and resource return

`CompletionStream.Close()` requests cancellation. `Wait()` can promptly return
a context error while the current batch is still running. `Released()` is a
separate channel: it closes only after all of the following have happened:

1. The request's last `PrefillBatch` call has returned. All parallel kernels in
   that call have joined, including when the engine is shutting down.
2. Its cancellation observation callbacks have stopped or been joined.
3. Its KV reservation and pages have been returned to the engine's page pool,
   and its admission permit has been returned.
4. Its terminal reply is published and its event channel is closed. Receiving
   this notification never requires draining the buffered token events.

For example, after abandoning a stream, wait on `stream.Released()` with a fresh,
bounded cleanup context; the original request context may already be canceled.
`Engine.Close()` waits for all accepted requests and the actor to drain, then
joins the engine's reusable compute workers before returning.

Returned pages can remain allocated in the pool for reuse. Caller-held event
and reply buffers also remain reachable. Release is an ownership guarantee,
not a claim that the GC freed those bytes or that process RSS fell. No resources
belonging to another request are included in a stream's release notification.

Individual request cancellation is checked around scheduling ticks. The shared
batch uses the engine context; it does not abort healthy peers when one request
cancels. Engine cancellation is checked between layers. A running matrix or
attention kernel is not interrupted. The [fixed worker executor](INFERENCE_WORKERS.md)
reuses kernel goroutines while retaining these cancellation points and joins;
the measurements do not establish a hard latency bound.

## Metrics

The following are fixed-size counters, gauges and histograms at `/metrics`.
They retain no per-request histories or unbounded label sets.

| Metric | Meaning |
| --- | --- |
| `monolith_released_total` | Accepted requests whose computation, observers, KV and admission ownership have ended, on every terminal path. |
| `monolith_admission_used` | Occupied admission permits, including prompt preparation, queueing, execution and cleanup. |
| `monolith_computing_sessions` | Sessions in the current `PrefillBatch` call; this is not a goroutine count. |
| `monolith_kernel_workers` | Compute worker goroutines owned by this engine, including idle workers. Zero for inline execution and after `Engine.Close()`. |
| `monolith_batch_compute_seconds` | Wall time inside a shared batch call, including allocation, scheduling and kernel joins; excludes token sampling and delivery. |
| `monolith_cancel_observed_to_compute_done_seconds` | Time from the first cancellation observation to the end of the last batch containing that request, clamped to zero if no computation followed the observation. |
| `monolith_cancel_observed_to_release_seconds` | Time from that same observation until the request's KV and admission ownership has ended and its observer callbacks have joined. |

An observation is the first of a synchronous stream `Close()` call, a
request-context cancellation callback, an engine-shutdown callback, or the actor
noticing cancellation. Context callbacks are scheduled asynchronously, so these
times do **not** measure network-disconnect-to-release latency, nor do they bound
time from the actual context signal. The trace records the observation source.
Callbacks use Go's [context.AfterFunc](https://pkg.go.dev/context#AfterFunc);
their stop functions alone do not join a running callback, so the engine
explicitly joins it before reporting release.

Each admitted request has at most two observation callbacks. Standard cancelable
contexts need no permanent waiting goroutine; each triggered callback does only
bounded local bookkeeping. Callback registrations are removed on success,
rejection, cancellation, slow-consumer failure and shutdown. Normal completion's
internal context cleanup is never recorded as a cancellation. A cancellation
racing with successful completion can still be observed before the terminal
decision, so observation counts need not equal `monolith_canceled_total`.

Both cancellation histograms receive one observation for each request with a
recorded cancellation, including queued requests. A queued request which never
computes contributes zero to the compute-tail histogram. Slow-consumer failure
alone is not a cancellation observation. Existing token-gap metrics measure
server sampling cadence across all requests; the load client can separate normal
readers, slow readers and planned cancellations for client-side distributions.

Stats are independently sampled atomics, not a transactional snapshot. In a
dedicated, quiescent server, active/queued/computing requests, admission permits
and KV reservations return to zero, and released equals submitted. Allocated KV
pages need not return to zero. Compute workers stay available while the engine
is open, so their count is not a request-drain condition.

## Execution traces

```sh
mkdir -p runs/lifecycle
./monolith serve -model runs/model.mglm -threads 4 \
  -max-batch 4 -queue 32 -kv-mib 64 \
  -trace runs/lifecycle/serve.trace
```

`-trace` requires a new file in an existing directory and never overwrites an
input or earlier trace. It captures the serving invocation after model loading;
on ordinary shutdown, the engine drains before the trace writer is stopped and
its output errors are checked. A shutdown timeout can leave requests without a
release event in the captured trace. The trace is process-wide and adds overhead;
use bounded diagnostic runs, since the file grows with runtime activity.

Each `monolith.request` task records enqueue attempts, admission, batch identifiers
and row counts, compute completion, cancellation observation/source, terminal
outcome, and release. A rejected submission ends with `not_enqueued`. Shared
`monolith.batch` and nested `monolith.prefill_layer` regions show CPU execution,
scheduling and GC in the runtime timeline. Task IDs and the engine-local batch
number relate requests to work; prompts, generated text and token IDs are not
added to trace annotations. Standard runtime traces still include runtime/code
metadata. Tasks are created for requests submitted while tracing is enabled.

After stopping the server:

```sh
go tool trace runs/lifecycle/serve.trace
# Noninteractive decoder check with named debug modes (Go 1.27):
go tool trace -d=parsed runs/lifecycle/serve.trace > runs/lifecycle/trace.txt
```

Older toolchains use `-d=1` instead; `go tool trace -h` lists supported modes.

All APIs used here are available in Go 1.22. FlightRecorder and synctest are not
required. Trace timing is diagnostic evidence, not an uninstrumented benchmark.

## Mixed-load validation

Use the [load client](tools/loadtest/README.md) against a dedicated server with a
checkpoint whose context fits both prompts plus the generation budget:

```sh
go run ./tools/loadtest -url http://127.0.0.1:8080 \
  -requests 36 -concurrency 8 -max-tokens 24 \
  -short-prompt 'the ' -long-prompt 'the cat rests. the cat rests. ' \
  -cancel-every 5 -cancel-after 2 -slow-every 3 -slow-delay 10ms \
  -drain-timeout 5s -details > runs/lifecycle/load.json
```

Check request outcomes and stream/final-response parity, healthy-reader token
gaps, `server_metrics.drain_status`, and the new histogram counts/sums. A planned
client cancellation may reach the server after generation finishes; it is not
automatically a server cancellation sample. Short output can fit into network
buffers without causing server backpressure. Neither this load nor a trace alone
proves a production capacity or latency guarantee.

Deterministic engine tests gate the real batch at its first layer after KV
allocation, cancel one active and one queued request, and prove that ownership
is retained until compute returns. The healthy peer matches independent
generation exactly. Further cases cover parent cancellation, shutdown, queued
cancellation without computation, unconsumed streams, late `Close`, permit reuse,
trace/output parity, output preservation and trace write failures.

A [local diagnostic receipt](benchmarks/serving-lifecycle-2026-10-15.json)
records one traced Apple M4 / Go 1.27.1 run with the 4,296,448-parameter `small`
fixture: four warmup requests, then 36 requests at concurrency eight. The measured
load completed 29 requests and canceled seven, with no protocol errors. All
accepted requests returned their reservations and admission permits. The decoded
trace contains 40 request tasks, 40 release/end pairs, seven cancellation
observations, 591 batch regions and 3,546 layer regions. Source, checkpoint and
local artifact hashes are included. The seven observed-cancellation-to-release
samples averaged 1.93 ms with tracing enabled; this small diagnostic sample does
not establish a latency bound, an uninstrumented result or a speed improvement.

```sh
GOMAXPROCS=4 go test -race -count=1 ./...
```
