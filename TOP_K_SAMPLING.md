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
