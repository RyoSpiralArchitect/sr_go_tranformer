# Concurrent inference diagnostics

The [single-row Q4 comparison](INFERENCE_PERFORMANCE.md#qwen25-7b-single-session-result)
found a 32.6% single-session improvement but a 7.1% slower two-session median.
Its paired kernel control had overlapping ranges. That evidence does not
identify the cause of the two-session change.

## Counterbalancing the cases

The comparison driver now supports `--case-order balanced`. It retains the
before/after/after/before process sequence and reverses session-case order in
the second process pair:

| Process | Runtime | Session cases, in execution order |
| ---: | --- | --- |
| 0 | before | 1, then 2 |
| 1 | after | 1, then 2 |
| 2 | after | 2, then 1 |
| 3 | before | 2, then 1 |

Each case retains one warmup and three measured trials, with fresh KV per
trial and a reused workspace per case. The fixed eight-token prefixes and
seven decode calls are unchanged. Each forward call's duration is now retained
as well as the seven-call sum. Timing bookkeeping occurs after the forward
interval; its seven-element array is allocated before allocation sampling.

The runner verifies actual execution order before sorting cases for numerical
comparison. Every compared prompt, model plan, logit/KV digest, generated ID
and workspace capacity must still match. Reports retain execution order and
per-process peak RSS. Default `--case-order fixed` preserves the earlier case
sequence; the added timing fields use version 2 of the receipt schema.

```sh
python3 -S scripts/compare_inference.py --model PATH_TO_QWEN_7B \
  --id qwen25-7b --baseline-ref 6b94640 --case-order balanced \
  --output-dir runs/qwen7b-balanced-diagnosis
```

This is a bounded comparison, with one process per variant/order combination
and three repeated trials within it. It does not eliminate background work,
thermal effects, nonlinear drift or carryover, and is not a confidence interval
or proof of a unique cause. Core-forward timing excludes model loading,
tokenization, hashing, sampling and HTTP delivery. A CPU-profiled or traced run
is a separate diagnostic and must not supply the reported speed comparison.

The standalone test probe also accepts `MONOLITH_PERF_CASE_ORDER=1` or `2` to
isolate a case for Go's test CPU profiling/tracing. It requires the same pinned
model/corpus inputs, `GOMAXPROCS=4`, a fresh `MONOLITH_PERF_OUTPUT` path and the
isolated build produced by the driver. Profiling the whole process includes
loading; interpret stacks accordingly instead of treating load time as decode.

## Qwen2.5-7B repeat on 2026-10-27

On Apple M4 / 24 GiB / Go 1.27.1, the isolated, cgo-free four-thread build
compared the PR31 runtime (`6b94640`) against PR32 (`1226102`). The new probe is
identical in both binaries. All four runs used the cached pinned
`Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28` with `q4-mixed` weights.

| Sessions | Before median / range, seconds | After median / range, seconds |
| ---: | --- | --- |
| 1 | 4.834 / 4.603–5.010 | 3.459 / 3.319–3.487 |
| 2 | 6.903 / 6.640–7.129 | 6.989 / 6.789–7.040 |

These are six seven-call decode sums per cell (three trials in each of two
processes), not independent model runs or HTTP throughput. The single-session
median is 28.4% shorter. The two-session difference is now +1.2%, with
overlapping trial ranges: the previous +7.1% slowdown was not consistently
reproduced. For order 1→2, two-session process medians were 7.082→6.862 seconds;
for order 2→1 they were 6.745→7.025 seconds. The difference changes sign, but
this sparse crossover cannot establish an order effect or another root cause.
The earlier negative result remains part of the evidence.

All prompts, generated IDs, logit/KV hashes, model plans and workspace capacities
match exactly. Workspace remains 3,033,344 / 6,066,688 bytes for one/two sessions.
Process peak RSS was 4.11–4.15 GiB before and 4.10–4.11 GiB after; no resident
memory reduction is claimed. Receipts retain per-step durations, process order,
model/build hashes, isolated effective Go settings and child resource usage:
[7B comparison](validation/performance/qwen7b-balanced-20261027.json) and
[360M protocol pilot](validation/performance/smol360-balanced-20261027.json).

## Locating the remaining work

Separate two-session CPU-profile and execution-trace processes used the PR32
binary above and the same pinned inputs. Their numerical signatures match the
unprofiled case; their times are excluded from the comparison table.

* The Q4 kernel accounts for 116.23 of 125.42 sampled CPU seconds on worker-run
  stacks (92.7% cumulative). These stacks include warmup, prefill and decode.
  Source-line samples identify the paired packed-weight loop as substantial
  work, alongside the four-row prefill loop. They do not measure decode alone.
* The trace reports 0.351 seconds of aggregate scheduler delay. Its much larger
  synchronization total includes necessary actor joins while kernels run,
  idle workers, the test harness and trace background tasks. Summed goroutine
  blocking time is not removable request latency or proof of channel overhead.
* Comparing the actual binaries also shows that the shared Q4 callback changed
  from a 112-byte to a 288-byte stack frame when the single-row helper was
  introduced. Thus unchanged two-row source is not proof of identical machine
  code. Frame size alone does not establish a performance cause.

This points to testing a complete-block specialization for the paired Q4 loop,
with the single-row path as a regression control, before changing worker
ownership or scheduling. It does **not** resolve a unique cause for the earlier
7.1% result. [Profile summaries](validation/performance/qwen7b-diagnostics-20261027.txt)
and the [diagnostic manifest](validation/performance/qwen7b-diagnostics-20261027.json)
retain the capture hashes, reports and interpretation limits. Raw profiles and
trace remain local under the ignored run directory; the manifest does not
pretend they are available from the public repository.
