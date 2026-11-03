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

By default, the engine actor copies committed complete 32-token pages from a successfully
completed session into its one-entry snapshot. Failed, canceled, and
slow-consumer sessions never publish a snapshot. A cache hit lends the matching
full pages read-only to the new session. It always leaves at least one input
token to forward, because it does not cache logits. The mutable suffix belongs
to the new session and can never write into the shared pages.

Add `-prefix-cache-tail` to `chat` or `serve` to retain committed positions between page boundaries. Full matching pages are still lent read-only. For a partial match the actor copies only the matching key and value rows into one private page from the new request’s reservation. This also works when two prompts diverge inside a previously full page. A one-token prompt always runs through the model, so it can produce logits.

The tail can retain up to one additional 32-token page within the configured prefix allowance. The allowance continues to bound snapshot storage when the last page is partial. Only recorded token IDs can match. A request that copies fewer than 32 matching tokens holds no snapshot reference after admission.

Snapshot replacement occurs with no outstanding borrowers, or when the
successfully finishing session is the current snapshot's sole borrower. The
engine actor waits for compute to join before copying completed suffix pages;
the shared prefix remains in place. This lets successive matching conversations
grow the retained prefix within the existing allowance. If another reader
remains, capture is skipped. A successful cacheable miss can still replace it.
Only one snapshot allocation is live, using existing page storage where possible,
without a background reclaimer or lock per page. This is not a multi-entry LRU.

The same model revision and reset epoch are required. Capture stops at the
cache's committed position, not the history length: the last sampled output
token may not have a KV row yet. Mutable suffix pages are copied, never taken
from the request's pool reservation. Cancellation, slow-consumer failure and
requests admitted before `/reset` cannot publish or extend retention.

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

- `prefix_hits` / `monolith_prefix_hits_total`: requests that reused at least one input token.
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

The [2026-10-30 pinned Mistral 7B receipt](validation/hf/mistral-prefix-cache-20261030.json)
records two 120-token chat prompts with 114 common token IDs, each generating
eight greedy output tokens. Reuse skipped 96 input positions (three complete
pages) on the second request. Both completions matched the disabled mode exactly;
computed input rows fell from 240 to 144 across the pair. The snapshot retained
24 MiB, and total allocated KV payload was 56 MiB within the 512 MiB engine budget.
Shutdown released the snapshot. Original model files matched the pinned hashes
before and after the run. The measured process peak RSS was about 3.94 GiB,
including model loading; it is distinct from the KV payload budget.

## Growing-prefix fixture

The [five-request native fixture](validation/performance/prefix-growth-fixture-20261030.json)
uses 66, 98, 130, 162 and 194 input positions with eight output tokens each.
Across 650 logical input positions it reuses 448 and computes 202. Retained
prefix pages grow from two to six, within the fixed allowance. FP32, Q8 and Q4
weight variants produce exactly the disabled-reference completions, including
a subsequent divergent branch. The [race-enabled log](validation/performance/prefix-growth-fixture-20261030.txt)
also records sole/multiple-reader ownership, uncommitted-token exclusion,
failure, deterministic cancellation and reset during a borrowed computation.

These are synthetic growing prompts, not a pretrained multi-turn chat or a
balanced speed measurement. The earlier Mistral receipt checks fixed-prefix
reuse; it does not establish the speed or quality of this extension policy.

## Five-turn Mistral chat comparison

The offline comparison runs five user messages through the same Mistral chat
template and streaming engine as interactive `chat`. Each next turn includes
the previous decoded assistant reply. Sampling is greedy with an eight-token
cap and natural EOS. The comparison checks the exact re-encoded prompt IDs,
every streamed token and text fragment as a joined reply, and the terminal
completion. The token histories and generated replies are saved with each trial.

```sh
python3 -S scripts/compare_prefix_chat.py --model PATH_TO_MISTRAL_7B \
  --output runs/mistral-five-turn
```

The runner needs both historical capture revisions in the local Git history.
It builds the current source and a control that replaces only the borrower guard
with its pre-growth implementation. The fixed binary has a separate test label
checked before the model loads. Each fresh process runs one conversation. The
six runs use disabled, fixed, growing, growing, fixed, disabled order with four
threads, context 512 and a total KV budget of 512 MiB. The two enabled modes
reserve 128 MiB within that total for the prefix.

First-token time starts before formatting the conversation and ends when the
caller receives the first token-ID event, which can carry empty text. Completion
time ends at the terminal reply. Model loading is outside both intervals.
Per-turn counts report actual input positions, reuse, and KV payload after
release; process peak RSS includes loading. The first turn starts with an empty
engine. The remaining turns keep that engine's snapshot. Every completed turn
is journaled, and a partial run remains marked incomplete.

### Five-turn results

The [complete receipt](validation/performance/mistral-five-turn-chat-20261101.json) records six processes, five turns per process. All modes had identical prompt IDs, delivered token IDs, display text and terminal completions at every turn. The table shows the median of two first-token times per mode, in seconds.

| Turn | Logical input | Disabled, s | Fixed, s | Growing, s | Reused tokens (fixed / growing) | Allocated KV MiB (disabled / fixed / growing) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56 | 22.652 | 23.042 | 25.887 | 0 / 0 | 16 / 24 / 24 |
| 2 | 80 | 42.712 | 19.832 | 30.057 | 32 / 32 | 24 / 24 / 32 |
| 3 | 103 | 46.745 | 32.429 | 19.465 | 32 / 64 | 32 / 32 / 40 |
| 4 | 126 | 51.276 | 42.610 | 16.271 | 32 / 96 | 40 / 40 / 48 |
| 5 | 155 | 62.139 | 55.172 | 12.583 | 32 / 128 | 48 / 48 / 56 |

Across 520 logical input positions, disabled/fixed/growing reuse skipped 0/128/320 positions. Final retained prefix payload was 0/8/40 MiB; maximum observed KV payload was 48/48/56 MiB within each 512 MiB engine budget. Prefix retention and live reservations returned to zero on shutdown. These byte counts describe KV payload. Per-process peak RSS in the receipt also includes model loading.

The first turn has an empty cache in every mode. Later turns use the actual decoded chat history and can cross different token/page boundaries. This is a bounded comparison on one shared Mac, with two trajectories per mode and a maximum of eight reply tokens.

A separate compiler was observed in 19 of 95 periodic [host snapshots](validation/performance/mistral-five-turn-host-20261101.json) across these six processes. All trials are retained. Token counts are deterministic; wall-clock timings include shared-host variation.

Across the two processes per mode, peak RSS including loading was 3.881–3.902 GiB disabled, 3.883–3.910 GiB fixed, and 3.909–3.910 GiB growing.

All measured inputs match commit `1791cf8`. The subsequent change to comparison-driver error reporting from #49 does not change the tested chat runtime or the isolated Go environment settings. Use the recorded commit to reproduce the exact inputs.

## Comparing exact tail reuse

Run the page-only and tail policies with the same production binary and local
Mistral v0.3 weights. Each fresh process runs one conversation, in page, tail,
tail, page order. Both policies use four Go threads and the same 512 MiB KV
budget with 128 MiB set aside for retention.

```sh
python3 -S scripts/compare_prefix_chat.py --comparison tail \
  --model PATH_TO_MISTRAL_7B --output runs/mistral-prefix-tail
```

Every turn records time through the first streamed token, the terminal reply,
actually computed and reused inputs, and KV payload after release. Each warm
latency sample sums turns 2–5 of one conversation. Compare the median of the
two sums per policy. The limits are at least 25% fewer computed input rows
across all five turns, at most one additional 8 MiB page at the observed KV
peak, and at least 10% less warm first-token time. The driver checks every
prompt, streamed reply and completion before applying those limits. A
completed disagreement retains the trial reports and skips ratios for unequal
trajectories. Interrupted runs leave their per-turn journals on disk.
