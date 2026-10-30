# Bounded top-k sampling

Non-greedy sampling retains at most k candidates when `0 < top_k <= eligible/4`,
where eligible excludes the blocked token if it is present. A heap keeps the
worst retained candidate at its root. Every vocabulary logit is still checked,
including masked IDs and candidates that cannot enter the heap. Repetition
penalties are applied before selection. Selected candidates are then sorted by
descending logit and ascending ID, just as in the full-sort reference.

The remaining temperature, top-p, probability accumulation and RNG operations
are unchanged. Ties and signed zero keep their original ID order. Invalid
distributions return before consuming randomness. Greedy sampling is unchanged;
top-k disabled or covering more than a quarter of eligible IDs uses full sort
to avoid an additional large heap pass. No sampling policy or precision changes.

Candidate storage in the bounded path is O(k), while validation still scans
the whole vocabulary. The history membership map remains: this is not an
allocation-free generator. Workspace, model weights and KV storage are unrelated
to these candidate-array savings. Sampler timings are not model throughput.

The frozen `8dca84e` sampler is retained in tests, including its optimized
greedy path. Tests compare selected sets, token IDs and complete RNG state
across ties, signed zero, extreme finite values, masked/nonfinite logits,
top-k and fallback boundaries, temperatures, top-p cutoffs, penalties, seeds,
32K/152K vocabularies, and seeded engine completions with growing prefixes.

```sh
GOMAXPROCS=4 ./scripts/go.sh test -run '^$' \
  -bench '^BenchmarkBoundedTopKSampling$' -benchmem -benchtime=250ms -count=3 -cpu=4
```

The [frozen protocol](validation/performance/top-k-protocol-20261030.json)
records both orders, a large-k fallback, disabled top-k, and a greedy control.
The [load client](tools/loadtest/README.md) now supports explicit non-greedy
cohorts; its default remains greedy for compatibility with earlier measurements.
Use the same prompt, seed and sampling options on both compared runtimes.

## Sampler measurement

Apple M4 / 24 GiB / Go 1.27.1, four threads, 256 history IDs, six samples per
variant (three in each order), temperature 0.8, top-k 40, top-p 0.95 and repeat
penalty 1.1. Medians, including the history map and all-logit validation:

| Vocabulary | Full sort, ms | Bounded top-k, ms | Allocated bytes before / after |
| ---: | ---: | ---: | ---: |
| 258 | 0.0155 | 0.0061 | 29,704 / 10,632 |
| 32,768 | 4.087 | 0.240 | 1,787,275 / 10,632 |
| 50,257 | 6.638 | 0.357 | 2,737,546 / 10,632 |
| 151,936 | 21.868 | 1.046 | 8,226,184 / 10,632 |

At 32,768 vocabulary entries, disabled top-k changed time by -0.05%, top-k
covering the full vocabulary by +1.47%, and the greedy control by +0.28%.
All frozen gates passed. Fallback candidate capacity now also accounts for an
absent blocked ID, avoiding the reference's final append growth. Greedy storage
is unchanged. The complete path still allocates; the history map is not removed.

[All 84 samples, gates and source hashes](validation/performance/top-k-sampling-20261030.json)
and [raw output](validation/performance/top-k-sampling-20261030.txt) are retained.
These are synthetic logit vectors, not a pretrained model or HTTP throughput
result. Full engine completion parity is checked separately under race detection;
no end-to-end speed or process-memory reduction is inferred from this table.
