# Selective prompt output heads

Direct generation and the inference engine skip final normalization and
vocabulary projection for prompt chunks whose logits will not be sampled.
The model layers and all committed KV values are still computed. A 275-token
prompt forwarded in 16-token chunks needs one output head instead of 18; this
is a head-call reduction, not an 18-fold end-to-end speed claim.

The internal `prefillSelected` call accepts a session-sized selection. Selected
final hidden rows are compacted before the head so mixed batches retain weight
reuse, then logits are expanded backwards to their original session slots.
Unselected rows are zeroed to prevent stale workspace contents from escaping.
A nil selection computes all heads. Public `PrefillBatch`, `DecodeBatch` and
`Step` continue returning every requested session's logits.

Direct generation selects only the final prompt chunk. The engine selects
sessions finishing their prompt in the current tick and sessions already
decoding. Sampling order, arithmetic, KV ownership and cancellation commits
are unchanged. Even a headless call checks cancellation before committing
cache positions. Existing workspace admission remains conservative: it still
reserves full session logits and the public result copy.

Tests compare all five-session selection masks, FP32/Q8/Q4 weights, one/four
workers, page crossings, workspace reuse, exact completions and invalid masks.
The retained all-head generation reference comes from `8dca84e` and already
has chunked prefill and request-owned workers. The
[frozen measurement protocol](validation/performance/selective-head-protocol-20261030.json)
separates a native large-vocabulary fixture from a pinned real-model probe.

```sh
GOMAXPROCS=4 ./scripts/go.sh test -run '^$' \
  -bench '^BenchmarkSelectiveHeadGenerate$' -benchmem -benchtime=250ms -count=3 -cpu=4
```

Speed and allocated bytes must be measured separately. The scratch bound and
weight/KV precision are unchanged; skipping heads does not establish lower
process RSS or improved model quality.

## Native measurement

On Apple M4 / 24 GiB / Go 1.27.1, four threads, the frozen single-layer
32,768-vocabulary fixture passed its gates. Medians of six samples per variant:

| Prompt / output tokens | All heads, ms | Selected heads, ms |
| --- | ---: | ---: |
| 3 / 1 | 5.070 | 2.326 |
| 33 / 1 | 6.918 | 3.723 |
| 129 / 1 | 21.829 | 9.918 |
| 257 / 1 | 42.021 | 19.005 |
| 3 / 32 | 58.687 | 54.163 |

This fixture has a disproportionately large output head relative to its one
model layer. The three-token path forwards one token at a time, so selection
also removes two heads from that short case. These numbers are not expected
7B speedups. The synthetic tokenizer renders output IDs directly and is not a
language/tokenizer compatibility test.

[All samples and source hashes](validation/performance/selective-head-native-20261030.json)
and [raw measurements](validation/performance/selective-head-native-20261030.txt)
are retained. The [initial probe failure](validation/performance/selective-head-initial-error-20261030.txt)
was a fixture vocabulary mismatch before timing; it was fixed before the
recorded measurements. No failed timing sample was discarded or rerun.

The real-model comparison uses the existing driver with the previous merged
runtime and a frozen 5% non-regression ceiling, rather than requiring a minimum
speedup from this small fraction of total 7B compute:

```sh
./scripts/compare_prompt_prefill.py --baseline 8dca84e --long-ratio-max 1.05 \
  --model PATH_TO_MISTRAL_7B --output runs/selective-head-real
```
