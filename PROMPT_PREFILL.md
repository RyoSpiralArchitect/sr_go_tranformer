# Bounded prompt prefill

Direct `Generate`, `GenerateChat` and CLI `generate` process prompts in groups
of at most 16 tokens, using the request's existing fixed workers and owned
workspace. Projection kernels can reuse each weight load across several rows,
and the output head is evaluated only for each chunk's final row. Generated
tokens still decode one at a time. Prompts of up to three encoded tokens keep
the previous tokenwise path.

The chunk planner estimates all floating-point scratch, including the
session-sized logits and reserved public copy, and attention scores rounded
to a KV page. It reduces the chunk to fit a soft 16 MiB prompt-scratch target.
A one-row fallback preserves requests whose minimum workspace exceeds that
soft target. The existing 512 MiB hard forward-workspace check remains in
force before KV allocation or model reads. Retained scratch can increase
relative to tokenwise prefill; this optimization targets time to first output,
not a blanket memory reduction.

Worker selection accounts for prompt-chunk work as well as single-row decode.
Workers and workspace remain exclusive to the invocation. Every kernel joins
before reuse or release. Cancellation is checked between chunks and layers;
a running kernel completes before cancellation returns. Chunking changes the
frequency of cancellation checkpoints, so cancellation latency is not claimed
to be identical to tokenwise forwarding.

## Exactness and ownership

Regression tests compare seeded sampling and greedy completions to the frozen
request-owned tokenwise path from `68ee4d8`. They cover FP32, Q8 and Q4 weights,
one/four workers, partial tiles, 16-token chunk boundaries and 32-token KV page
boundaries. A separate comparison checks every committed KV value and each
chunk's final logits exactly against tokenwise forwarding. HF chat parity,
EOS, generated-token cancellation and shared-model concurrency remain covered
by the generation tests. Prompt cancellation before sampling and clean retry
are tested with active background workers.

The prompt planner is also checked against an independently enumerated set of
forward arrays for large geometry-only fixtures, without allocating weights.
Training, evaluation, engine scheduling and the public `PrefillBatch` API are
unchanged. Interactive chat/serving already use their engine's chunked prefill;
this change covers the direct-generation path.

## Performance protocol

The [frozen protocol](validation/performance/prompt-prefill-protocol-20261030.json)
compares both implementation orders, three samples per order, with four
threads. The native fixture has dimension 256, hidden width 704, one layer,
33/129/257-token prompts and a three-token short control. Each invocation
returns one generated ID, so the measurement includes prompt preparation,
prefill, sampling, worker startup/teardown and KV release. The reference already
owns reusable workers and workspace; this isolates prompt grouping from the
previous worker-reuse improvement.

```sh
GOENV=off GOTOOLCHAIN=local GOWORK=off GOFLAGS= GOEXPERIMENT= \
  GOPROXY=off GOSUMDB=off CGO_ENABLED=0 GOMAXPROCS=4 \
  ./scripts/go.sh test -run '^$' -bench '^BenchmarkPromptPrefillGenerate$' \
  -benchmem -benchtime=250ms -count=3 -cpu=4
```

Adoption requires exact output/KV parity, at least 5% less median time at
129/257 prompt tokens, and no more than 5% regression for the three-token
control. Native fixture results do not establish pretrained 7B performance;
real-model comparisons must be recorded separately. All samples, including
regressions, are retained.

## Native first-token measurement

Apple M4 / 24 GiB / Go 1.27.1, four threads, three 250 ms samples in each
implementation order. This native fixture is not a complete pretrained model.

| Prompt tokens | Before median, ms | Chunked median, ms | Allocated bytes before / after |
| ---: | ---: | ---: | ---: |
| 3 | 0.936 | 0.947 | 78,996 / 78,991 |
| 33 | 10.617 | 2.319 | 140,600 / 343,424 |
| 129 | 43.814 | 8.525 | 332,194 / 454,578 |
| 257 | 88.887 | 18.129 | 589,594 / 610,601 |

The frozen timing gates passed. Larger retained scratch increases allocated
bytes for some shapes even as forwarding and dispatch overhead fall. All raw
samples and source hashes are in
[the receipt](validation/performance/prompt-prefill-20261030.json) and
[raw output](validation/performance/prompt-prefill-20261030.txt).
The optional `scripts/compare_prompt_prefill.py --model PATH --output RUN_DIR`
builds baseline68ee4d8 and the current runtime with an identical real-model
first-token probe, checks original source files before/after, and uses ABBA
process order with reversed case order in the second pair. It writes its fixed
protocol before measurement. Model loading is outside the Generate interval;
encoding and sampling are included. No pretrained-model speed claim is made
from the native measurements above.
