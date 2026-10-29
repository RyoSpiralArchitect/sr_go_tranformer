# Engine prefix cache

Interactive `chat` and HTTP `serve` can retain one completed prefix per engine.
The cache is opt-in: `-prefix-cache-mib 0` (the default) keeps the existing
behavior. It reuses only exact token-ID prefixes from the same model revision.
Templates are applied before matching, so a changed template prefix, whitespace,
message order, or re-encoded assistant text can reduce or eliminate reuse.

```sh
HF_HUB_OFFLINE=1 ./monolith infer local \
  -path "$HOME/models/mistral7b-instruct-v0.3" -context 1024 \
  -weight-format q4-mixed -weight-mib 8192 \
  chat -kv-mib 512 -prefix-cache-mib 128 -max-tokens 64 -threads 4

HF_HUB_OFFLINE=1 ./monolith infer local \
  -path "$HOME/models/mistral7b-instruct-v0.3" -context 1024 \
  -weight-format q4-mixed -weight-mib 8192 \
  serve -kv-mib 512 -prefix-cache-mib 128 -max-batch 2 -threads 4
```

Build first with `./scripts/go.sh build -trimpath -o monolith .`. Prefix caching
applies to the engine paths; one-shot `generate` does not retain requests.
Model weights and the tokenizer must remain immutable during engine use.

## Ownership and memory

The engine actor copies committed complete 32-token pages from a successfully
completed session into its one-entry snapshot. Failed, canceled, and
slow-consumer sessions never publish a snapshot. A cache hit lends the matching
full pages read-only to the new session. It always leaves at least one input
token to forward, because it does not cache logits. The mutable suffix belongs
to the new session and can never write into the shared pages.

Snapshot replacement occurs only with no outstanding borrowers, using existing
page storage where possible. A request borrowing the current snapshot therefore
does not extend it on completion. A successful cacheable miss can replace it.
This conservative policy permits only one live snapshot allocation, without a
background reclaimer, lock per page, or unbounded generations of old prefixes.
It is not a multi-entry LRU or a complete conversation cache.

`-prefix-cache-mib` is carved out of `-kv-mib`, not added to it. The mutable pool
uses the remainder. Full request reservations still use the complete prompt
plus output budget, including borrowed pages, so enabling reuse does not
increase admission capacity. Reserve enough mutable space for a full request.
For example, Mistral v0.3 has 8 MiB FP32 KV pages; 128 MiB of prefix allowance
can retain up to 512 token positions. A nonzero allowance must fit at least one
page and leave space for one mutable page. Prefix and mutable limits are rounded
down separately; unused sub-page bytes cannot be borrowed across the split.

The budget covers KV float payload. Token IDs, page metadata, model weights,
workspace, runtime allocation overhead, and memory awaiting garbage collection
are additional. Retained spare snapshot pages remain charged after replacement
with a shorter prefix. The existing model context and workspace limits apply.
KV precision and weight quantization are unchanged.

## Visibility and invalidation

With the option enabled, interactive chat reports cumulative hits, input tokens
reused, and snapshot bytes after each successful turn. `/reset` clears both text
history and prefix lookup. `Engine.ClearPrefix(ctx)` provides the same actor
operation to Go callers. Existing readers finish with their immutable pages;
the final reader releases an invalidated snapshot. Sessions admitted before a
clear cannot republish the old history. `Engine.Close()` joins work and releases
the snapshot and all borrower references.

`Engine.Stats()` and HTTP `/metrics` expose:

- `prefix_hits` / `monolith_prefix_hits_total`: requests that reused full pages.
- `prefix_tokens_reused` / `monolith_prefix_tokens_reused_total`: input positions
  skipped through reuse.
- `prefix_bytes` / `monolith_prefix_bytes`: retained snapshot payload, including
  pages still borrowed after invalidation.

`kv_allocated_bytes` includes both mutable-pool and snapshot payload;
`kv_reserved_bytes` counts mutable request reservations. `input_tokens` counts
actually computed input rows; reused rows are reported separately. Completion
usage continues to report the entire logical prompt length.

The snapshot retains token IDs and KV after a request completes. Enable it only
for an engine whose callers are allowed to share this retention and timing
behavior. Use separate engines for separate trust boundaries. No cache content
is serialized, shared with other engines, or matched by text alone.

## Validation

Race-enabled tests check exact logits, KV rows and completions, divergent tails,
page identity, final-token recomputation, model/revision mismatch, attempted
rollback into immutable pages, replacement with live borrowers, cancellation,
slow-consumer failure, reset, concurrent requests and shutdown. Budget and CLI
validation cover negative, equal-to-total, sub-page and oversized allowances.

A manual real-model check uses a single local Mistral Q4-mixed model and two
chat prompts with a common prefix and different suffixes. Disabled and enabled
modes must produce identical completions, while the enabled second request must
report at least 64 reused input tokens. Original model files should be verified
against the pinned catalog before and after the run.

```sh
HF_HUB_OFFLINE=1 GOMAXPROCS=4 \
  MONOLITH_PREFIX_MODEL=PATH_TO_MISTRAL_7B \
  MONOLITH_PREFIX_OUTPUT=runs/prefix-real.json \
  ./scripts/go.sh test -run '^TestHFPrefixCacheReal$' -v -timeout 13m
```

This is a functionality/equivalence probe, not a language-quality evaluation or
a speed benchmark. The single before/after pair has no balanced timing control;
its durations must not be presented as an established throughput gain.
