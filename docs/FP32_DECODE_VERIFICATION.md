# FP32 decode kernel verification record

Verification-only document. It consolidates the evidence behind the FP32
remainder-row input-sharing optimizations (PRs [#58](https://github.com/RyoSpiralArchitect/sr_go_transformer/pull/58)
and [#59](https://github.com/RyoSpiralArchitect/sr_go_transformer/pull/59)) and the
rejected Q8 analogue. Nothing here is a universal performance claim: every
measurement is bound to the recorded host, toolchain, fixture or model, and
slower or neutral observations are kept alongside favourable ones.

## What changed

`linearFP32Range` splits each task tile into a four-sequence-row path and a
remainder-row path. Only the remainder-row path changed, in two steps:

1. **PR #58** (`5a5f87f`): two output accumulators share each input load.
2. **PR #59** (`511a5cc`): extended to four accumulators, with two-column and
   scalar loops covering column tails.

Each output keeps its original ascending input-index accumulation order, so
results are bit-identical to the pre-change kernel. Task partitioning, worker
counts, parallel thresholds and the four-row path are unchanged.

## Numerical verification

- Frozen scalar reference compared by `math.Float32bits` over 630 small shape
  combinations plus larger serial/parallel calls and special FP32 bit
  patterns (`cmd/monolith/fp32_outputs_test.go`,
  `cmd/monolith/fp32_outputs4_test.go`).
- Zero warm allocations and no post-join retained invocation state.
- Every completion compared byte-for-byte in kernel, synthetic-generation and
  real-model runs; no divergence observed.

## Timing evidence (Apple M4, Go 1.27.3)

### Kernel level (three fresh processes each, alternating order)

| Stage | workers1 decode shapes | workers4 decode shapes |
|---|---:|---:|
| Paired vs original scalar | ~0.51–0.54x | ~0.52–0.81x |
| Quad vs paired | ~0.62–0.65x | ~0.63–0.76x |

Unchanged four-row controls stayed near 1.00; some mixed-row conditions
observed ~1% slower timings and are retained in the raw receipts.

### Whole generation, synthetic FP32 fixture (alternating, completions identical)

| Shape | paired vs original | quad vs paired |
|---|---:|---:|
| prefix33 / outputs128 | ~0.68x | ~0.78x |
| prefix33 / outputs8 | ~0.79x | ~0.87x |
| long-prefill / outputs1 | ~0.97–0.99x | ~0.99x |

Long-prefill single-token cases show much smaller differences: decode-time
remainder rows are a small share there.

### Real model: Llama-3.2-3B-Instruct, f32 weights, greedy, threads=4

Whole-process wall clock (`/usr/bin/time -p`), alternating fresh processes,
byte-identical outputs; includes weight load, dequantization, prefill and
decode.

| Run | reference | candidate | ratio |
|---|---:|---:|---:|
| 32 tokens (median of 3) | 32.45s | 21.60s | 0.666 |
| 64 tokens | 54.89s | 34.91s | 0.636 |
| 128 tokens: paired baseline vs quad | 59.09–59.62s | 42.75–44.25s | ~0.73 |
| 256 tokens | 191.25s | 117.59s | 0.615 |

Estimated decode-only cost (32 vs 64 token delta): ~0.59x reference.

### Q8 control

- Q8 weights route through `linearQ8Into`/`linearQ8WideInto`/
  `dotQ8PairBlock64`, not `linearFP32Range`: measured ratio ~1.00 — a
  no-regression control, not an improvement claim.
- Applying the same column-pairing to `linearQ8Into`'s single remainder row
  measured ~1.02–1.05x (slower) at workers1 across three processes and was
  **rejected**. The bit-exact experiment remains in
  `cmd/monolith/q8_outputs_test.go`.

## Receipts

- `validation/performance/fp32-outputs-selection-20261107.json` (+ kernel,
  generation and control run files)
- `validation/performance/fp32-outputs-realmodel-20261107.json`
- `validation/performance/fp32-outputs4-selection-20261107.json` (+ kernel and
  generation run files)
- `validation/performance/q8-outputs-kernel-20261107-run{1,2,3}.json`
- Raw harnesses and binaries: `runs/` (ignored)

## Limitations

- Single real model, single prompt family, greedy sampling, one Apple M4
  host. Not a serving-throughput, tail-latency or memory-footprint claim.
- No claim for q8/q4 kernels, other platforms or toolchains, other models, or
  prefill-dominated workloads.
- The Q8 rejection is a measured outcome on this kernel structure, not a
  statement that quantized decode cannot be improved by other means.
