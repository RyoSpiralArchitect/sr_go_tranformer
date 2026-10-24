# Inference performance

Measure token selection separately from model projection, attention, source
loading and HTTP delivery. A faster sampler does not imply the same speedup
for end-to-end generation. Weights remain FP32.

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
