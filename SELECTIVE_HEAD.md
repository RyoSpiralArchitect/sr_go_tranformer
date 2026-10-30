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
