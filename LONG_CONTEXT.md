# Explicit KV budgets and long HF contexts

HF model metadata validation and runtime KV admission are separate. Supported
HF models may describe contexts through 32,768 positions, while each execution
must have an explicit finite budget for the pages it can allocate. Native
training, checkpoint loading and evaluation retain their original allocation
limits. Weight formats, KV precision, RoPE and attention arithmetic are unchanged.

Direct generation and token-ID probes default to **512 MiB of KV pages**.
The `-kv-mib` action flag increases or reduces that budget explicitly; it never
changes automatically with the requested context. The direct HF loader rejects
an underfunded selected context before loading weight tensors. Page storage is
allocated lazily during execution. Weights, activations, tokenizer data and Go
runtime overhead are additional, so this is not a process RSS limit.

## Plan before loading

For an existing original Mistral-7B-Instruct-v0.3 Safetensors folder:

```sh
HF_HUB_OFFLINE=1 ./monolith infer local \
  -path "$HOME/models/mistral7b-instruct-v0.3" \
  -context 8192 -weight-format q4-mixed -weight-mib 8192 \
  check -chat -kv-mib 2048 -json
```

`kv_plan` reports the selected positions, page count, bytes per page, required
page-rounded bytes, budget bytes and whether the context fits. An insufficient
budget returns an error stating required and available bytes. `check` inspects
metadata and tokenizer compatibility; it does not load or validate weight
payload values and does not establish generation quality.

The pinned Mistral v0.3 profile uses 32 layers and 1,024 KV values per token per
K/V tensor. At FP32, KV storage is 256 KiB per position and 8 MiB per 32-token
page. The selected full-context payload is therefore:

| Context positions | Pages | KV payload |
| ---: | ---: | ---: |
| 2,048 | 64 | 512 MiB |
| 4,096 | 128 | 1 GiB |
| 8,192 | 256 | 2 GiB |

These are storage calculations, not observed RSS or speed measurements.
Non-page-aligned limits round up; even a one-token request needs one page.

## Generate within that budget

```sh
HF_HUB_OFFLINE=1 ./monolith infer local \
  -path "$HOME/models/mistral7b-instruct-v0.3" \
  -context 8192 -weight-format q4-mixed -weight-mib 8192 \
  generate -chat -prompt 'Explain the supplied passage.' \
  -max-tokens 128 -temperature 0 -threads 4 -kv-mib 2048
```

`-kv-mib` is an **action flag**, after `check`, `generate`, or `ids`. Source and
HF sizing flags remain before the action. Budgets accept 1–1,048,576 MiB;
invalid values are rejected before source acquisition. Fetch and inspect remain
metadata/source operations and do not reserve an execution budget.

Chat and serving retain their existing shared `-kv-mib` page-pool budget and
full-request reservations. Their pool may be smaller than the model's selected
maximum context: requests are admitted only when their complete page
reservation fits. A large context in metadata does not promise that every
request is admitted. There is no implicit escalation of pool size.

## Ownership and programmatic use

`NewCache()` captures the model's immutable direct budget, defaulting to
512 MiB. `NewCacheWithBudget(bytes)` provides explicit full-context admission
for a caller-owned cache and does not mutate the model or another cache's
limit. `Generate` and the token-ID probe check the complete requested token
budget before starting prompt work. `PrefillBatch` checks all direct cache
page requirements before allocating workspace or KV pages or modifying any
session. Pooled caches continue to use their existing reservation accounting.

Shape and byte limits are validated before multiplication. The workspace
budget remains independent of KV storage. Increasing KV capacity therefore
does not remove the 512 MiB inference workspace guard or native training limits.

## Verification scope

Regression tests verify 4K/8K Mistral storage arithmetic without allocating
large weights, rejection before weight loading, page rounding, invalid bounds,
all-session preflight before mutation, cancellation, independent cache budgets,
pooled versus direct admission, and unchanged native model limits. A tiny
HF-format fixture actually executes all 8,192 positions and checks the complete
cache/page tail and context-overrun rejection under the race detector.

The tiny fixture establishes execution/admission mechanics. It is not evidence
of 7B long-context quality or full-8K throughput. Any larger-model receipt must
state its actual prompt/generated lengths independently of the configured
maximum context.
