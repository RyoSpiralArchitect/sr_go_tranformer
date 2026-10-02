# Streaming HTTP load test

This standard-library Go client exercises `/v1/completions` with a bounded
worker pool and emits a JSON report. Odd-numbered requests use the short
prompt; even-numbered requests use the long prompt. Cancellation and slow
reading are selected independently by request number.

Run a server with a checkpoint, then run the client from the repository root:

```sh
go run ./tools/loadtest \
  -url http://127.0.0.1:8080 \
  -requests 60 -concurrency 4 -max-tokens 16 \
  -short-prompt 'the ' \
  -long-prompt 'the cat rests. the cat rests. the cat rests. the cat rests. ' \
  -cancel-every 7 -cancel-after 2 \
  -slow-every 5 -slow-delay 50ms \
  -timeout 30s -sample-interval 100ms \
  > loadtest.json
```

Use `./scripts/go.sh run ./tools/loadtest` if using the repository's local Go
toolchain. Both prompts plus the generation budget must fit the checkpoint's
context. The defaults request at most 16 generated tokens. No request is retried.
Requests use greedy sampling (`temperature: 0`, `repeat_penalty: 1`).

The number of in-flight requests is bounded by `-concurrency`. A separate
connection fetches `/metrics` before the workload, periodically during it, and
after all client requests finish. There is at most one metrics request in flight.
`-details` includes individual request results, sorted by request number; prompt
contents and generated text are omitted from the report.

## Reading the report

- `outcomes` distinguishes completed requests, planned client cancellations,
  overload responses, request errors, server errors, protocol errors, timeouts,
  interruptions, server cancellations, and transport failures. Server deadline
  expiry is classified as a timeout in both JSON and SSE responses. SSE errors
  include their server code.
- `workloads` separates short and long prompts, normal and slow readers, and
  planned cancellations. Completed latency distributions include only requests
  that completed and passed stream validation. `all_observed_first_token_event`
  also includes requests that subsequently failed or were canceled.
- `completed_first_token_event` measures from sending the HTTP request to the
  first parsed **token event**, including queueing and prefill. It does not measure
  receipt of response headers or the first nonempty piece of text. UTF-8 byte
  fragments and special tokens may produce empty text events.
- `completed_token_event_gap` measures time between client-observed token
  events. Network buffering and the selected reader delay affect this number;
  it is not a direct measurement of server compute time. Each token gap is an
  observation, so longer responses contribute more observations.
- `completed_latency` includes consuming the terminal event and stream EOF.
  `elapsed_seconds` covers the client workload and excludes the initial and
  final metrics fetches. Completed token events per second use that elapsed
  time; this aggregate includes the configured mixture of readers and cancellations.
- Quantiles use the nearest-rank method. Each distribution retains up to
  100,000 observations, with deterministic reservoir sampling beyond that limit.
  `quantiles_sampled` indicates approximation. Counts, means, minima, and maxima
  still include every observation. A distribution with no observations is `null`.
- `server_metrics.sampled_peaks` contains sampled Go heap, goroutine, active
  request, queue, and KV usage maxima. These are **sampled peaks**, not guaranteed
  instantaneous maxima. Go heap allocation and heap reservation are **not RSS**.
- GC and request counter deltas use the before/after snapshots. They include
  any other traffic to the same server. Missing metrics and counter resets are
  reported explicitly. Sampling itself adds HTTP and runtime-observation work.

The client checks that token IDs, their count, and concatenated event text match
the terminal `done` event. It rejects truncated streams and events after `done`.
An intentional cancellation stops validation at the selected token event.
Canceling a client request does not prove that the server still had generation
work outstanding when it received the cancellation; inspect server counters and
KV measurements as well.

Slow readers sleep between parsed token events. Short responses may fit entirely
inside the network buffers, so this workload alone does not establish that
server-side backpressure occurred. Exercise larger outputs or use deterministic
engine tests to validate that condition.

Exit status is zero for completed requests, planned cancellations, and overload
responses. Other request outcomes return status 1 after writing the report;
invalid options return status 2. Metrics failures are recorded in the report and
do not change the request-outcome exit status. Use `-help` to list all options.

## Tests

```sh
go test -race ./tools/loadtest
```

Tests cover SSE framing, text/token parity, empty UTF-8 fragment events, premature
EOF, cancellations, timeouts, outcome classification, mixed-workload grouping,
worker bounds, metric deltas, and distribution calculations.
