# Long generation

Long decode repeatedly reads a growing KV prefix. The inference attention
kernel now partitions independent row/head tasks across the engine's existing
fixed workers. A single session can therefore use multiple workers when its
attention workload exceeds the existing parallel threshold. Each head keeps
the original dot-product, normalization and value-accumulation order.

A lane is a disjoint range of head tasks, not another goroutine pool. Each lane
owns one score slice and reuses it between heads. Jobs join before the actor
reuses scratch, writes another layer or commits cache positions. Engine
cancellation retains the existing layer/commit checkpoints; running kernels
still finish and join. Training and evaluation arithmetic are unchanged.

## Bounded scratch

For `rows` input rows, `heads` query heads, `workers` configured workers and
`maxPos` largest causal prefix, attention uses at most
`min(workers, rows * heads, 256)` lanes. Small kernels use one lane. Standalone
model calls derive the worker count from GOMAXPROCS; engines keep their fixed
executor. The score stride rounds `maxPos` up to a 32-token KV page, capped at
the model's selected context limit.

Score storage is therefore `lanes * stride * 4` bytes instead of
`rows * maxPos * 4`. One lane never shares writable scratch with another.
Existing capacity accounting includes this storage and retained capacities
under the same 512 MiB workspace budget, plus the reserved public output copy.
There are no expanded attention probabilities or extra KV copies.

This can reduce prefill scratch substantially. For a hypothetical admitted
1,024-row / 32,768-position shape at four workers, the score term is 512 KiB
instead of 128 MiB; other workspace and model/KV limits still apply. Conversely,
a single long decode uses up to four score slices instead of one. This is a
bounded tradeoff, not a claim that every shape uses less scratch.

When growth is needed, a fixed lane layout allocates scores at most once per
32-token page rather than every new position. Retained buffers from earlier
prefill may already be large enough; total allocation and RSS effects must be
measured for the actual request. Model weights, quantization and KV precision
are unchanged.

## Attention-only measurements

Apple M4 / 24 GiB / Go 1.27.1, four Go threads/workers, three 150 ms samples in
each implementation order. The synthetic fixture uses Mistral's 4,096-wide
queries, 32 query heads and 8 KV heads, with initialized paged KV for one layer.
All rows share a read-only cache, as rows of one prefill chunk can; causal
positions vary across rows. It is not an execution of the complete 7B model.

| Prefix positions | Rows | Before, ms | Lanes, ms | Time reduction |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 0.544 | 0.171 | 68.6% |
| 128 | 2 | 0.576 | 0.329 | 42.8% |
| 128 | 16 | 2.429 | 2.439 | -0.4% |
| 2,048 | 1 | 11.932 | 5.016 | 58.0% |
| 2,048 | 2 | 12.333 | 6.423 | 47.9% |
| 2,048 | 16 | 52.817 | 54.397 | -3.0% |
| 8,192 | 1 | 88.266 | 28.482 | 67.7% |
| 8,192 | 2 | 93.592 | 59.430 | 36.5% |
| 8,192 | 16 | 421.517 | 429.363 | -1.9% |

These are medians, not confidence intervals. Six samples per variant/case and
all regressions are retained in the [raw output](validation/performance/attention-lanes-20261030.txt)
and [summary](validation/performance/attention-lanes-20261030.json). Already
parallel 16-row controls do not establish a speedup. Desktop activity and
thermal state were uncontrolled. No concurrent heavy tests, builds or profiler
ran during the timed kernel/model probes. The 8,192-position microbenchmark
does not establish that a full 7B model is admitted at that context.

```sh
GOENV=off GOTOOLCHAIN=local GOWORK=off GOFLAGS= GOEXPERIMENT= \
  GOPROXY=off GOSUMDB=off CGO_ENABLED=0 GOMAXPROCS=4 \
  ./scripts/go.sh test -run '^$' -bench '^BenchmarkAttentionLanes$' \
  -benchmem -benchtime=150ms -count=3 -cpu=4
```

## Real-model comparison protocol

The [frozen plan](validation/performance/long-generation-protocol-20261030.json)
uses pinned local Mistral-7B-Instruct-v0.3, Q4-mixed, context 1,024 and four
threads/workers. The original model files are checked before and after the
run. Baseline `7167259` and candidate use identical standalone test harnesses
and isolated Go settings, with one resident model at a time.

Process order is before/after/after/before. Each process runs a short control
(8-token prefix, 8 output IDs, one warmup and three measured trials) and a long
case (256-token prefix in chunks of 16, then 256 output IDs). Case order is
reversed in the second process pair. Each variant supplies two complete long
trajectories and six short trials. There are 255 timed long decode calls, and
511 committed KV positions at completion. Prefill and decode are timed
separately. Source loading, tokenization, sampling and hashing are outside
these core-forward intervals. Process peak RSS includes the whole process.

This fixed-step probe continues at EOS so both runtimes perform the same full
workload. It is not a natural stopping-length or language-quality evaluation.
All produced logit and used-KV digests, generated IDs, model plans, KV allocation
and non-score workspace capacities must match exactly. Only score scratch may
change; its size and growth are checked independently. The adoption gate
requires at least 5% shorter long decode and no more than 5% short-control
regression, in addition to the attention-only and exact-output gates.

```sh
python3 -S scripts/compare_long_generation.py \
  --model PATH_TO_MISTRAL_7B --id mistral-7b-v03 \
  --baseline-ref 7167259 --output-dir runs/mistral-long-comparison
```

## Measured Mistral 7B result

Candidate `a459cb4` passed the frozen adoption gates. The full
[receipt](validation/performance/mistral7b-long-generation-20261030.json)
contains model/source/build hashes, every per-token interval, output IDs,
logit/KV digests and per-process resource measurements. The
[acceptance summary](validation/performance/long-generation-acceptance-20261030.json)
retains the frozen plan and aggregated checks.

| Core forward interval | Before median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| Short decode, 7 calls | 3.175 s | 3.221 s | 1.5% longer |
| Long decode, 255 calls | 139.815 s | 126.877 s | 9.3% shorter |
| Long prefill, 256 tokens | 91.069 s | 90.849 s | 0.2% shorter |

Long decode ranges were 138.703–140.926 s before and 124.749–129.005 s after.
All output IDs, logit/KV digests, KV bytes and non-score workspace capacities
matched across all four processes. Long-case retained scores were 16,384 B
before and 8,192 B after; total workspace was 4,341,760 B and 4,333,568 B.
The candidate grew score storage eight times across 255 decode calls, within
the page-bound gate. The old buffer retained the larger prefill allocation and
did not grow. Short-case scores increased from 256 B to 512 B.

Process peak RSS was 4,360,060,928–4,380,049,408 B before and
4,376,576,000–4,379,574,272 B after. These overlapping ranges do not establish
lower total process memory. Weight payload and long-case KV allocation
(134,217,728 B) were unchanged. The main measured gain is reduced long-decode
wall time, at the cost of more parallel CPU work. There are only two long
trajectories per variant on this host; the short control's drift and the
16-row microbenchmark regressions are retained. This is not an HTTP latency,
concurrent-serving throughput, or broad language-quality result.

Correctness checks include exact per-head fixtures, page-growth bounds,
consecutive decode across the attention parallel threshold, cancellation,
workspace admission, and race tests. No quantization policy or output quality
tradeoff was introduced by this attention change.
