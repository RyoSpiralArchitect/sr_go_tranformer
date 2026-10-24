# Choose where inference runs

Training and inference have separate entry points:

| Entry point | Execution | Model files |
| --- | --- | --- |
| `monolith learn` | Native Go training, evaluation and data preparation | Native checkpoints |
| `monolith infer local` | Go CPU kernels and bounded serving engine | Existing files, or automatic HF download and cache |
| `monolith infer remote` | Hugging Face provider or dedicated endpoint | No local weights or tokenizer |

Backends never switch automatically. HF imports remain inference-only; they
cannot be trained or saved as native checkpoints. Existing direct commands
remain aliases for compatibility.

## Local execution with automatic download

Put source and memory flags **before** the action, and generation flags after:

Action flags, sampling bounds, input conflicts and engine capacity are checked
before contacting HF or creating a cache. Preflight uses the same parser as
execution and does not read prompt files/stdin or change worker settings.
Model-dependent compatibility and context checks follow source resolution.

```sh
./monolith infer local -repo HuggingFaceTB/SmolLM2-135M-Instruct \
  -revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  -context 512 -weight-mib 600 \
  generate -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 32 -temperature 0 -repeat-penalty 1
./monolith infer local -repo HuggingFaceTB/SmolLM2-135M-Instruct \
  -revision 12fd25f77366fa6b3b4b768ec3050bf629380bac -offline chat
./monolith infer local -dir runs/qwen25 check -chat
./monolith infer local -checkpoint runs/demo.mglm generate -prompt 'the spiral'
```

The default revision is `main`. Online resolution obtains an immutable commit
from HF; pin a commit explicitly for reproducible runs. The default cache is
`sr_go_tranformer/hf` inside the OS user cache. Use `-cache-dir` to change it.
`-offline` performs no network requests and requires a previously verified
snapshot (and a recorded reference when using a branch/tag).

`HF_HUB_OFFLINE=1` also enables offline resolution. Following the
[HF boolean convention](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#boolean-values),
`1`, `ON`, `YES` and `TRUE` enable it, case-insensitively; all other values
disable it. `-offline=false` cannot override an enabled environment variable.
Cached snapshots are still fully hash-checked. A missing or corrupt snapshot
fails without a metadata request, download or automatic repair.

```sh
HF_HUB_OFFLINE=1 ./monolith infer local \
  -repo HuggingFaceTB/SmolLM2-135M-Instruct \
  -revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  -context 512 -weight-mib 600 generate -prompt 'Hello' -max-tokens 8
```

This application also rejects `infer remote models`, `generate` and `chat`
while the environment variable is enabled, before waiting for input or making
HTTP requests. `infer models` (the built-in guide), help, existing `-dir`
models, native `-checkpoint` models and `learn` remain available. The local
cache is this application's verified cache, not the Python HF cache layout.

Downloads use only standard Go networking. No HF CLI, Python, Git checkout,
pickle loading or repository code execution is involved. Config and tokenizer
compatibility are checked before weight downloads. Source files are bounded by
`-download-mib` (default 2048, maximum 16384), verified against HF Git blob or
LFS identities, and published atomically only when complete. SHA-256 receipts
are verified on every cache reuse. Interrupted staging files are discarded.
Different processes can safely publish the same immutable snapshot.

Public repositories need no token. Set `HF_TOKEN` in the environment for access
to repositories your account can read. Tokens are not saved to the cache and
are removed on cross-host download redirects. Model access and licenses remain
the responsibility of the caller.

The download budget is separate from `-weight-mib` (FP32 resident weights),
`-context`, and serving `-kv-mib`. None is a process RSS limit. See
[local HF compatibility](HF_INFERENCE.md) for supported profiles and validation.
Available local actions: `check`, `inspect`, `tokens`, `ids`, `generate`, `chat`,
and `serve`. Native checkpoints accept `inspect`, `generate` and `serve`.

## Remote execution without downloading weights

Set an HF token with inference access in your shell's environment. Remote
generation may incur provider charges under your HF account.

```sh
# Live availability; no token is required for this public listing.
./monolith infer remote models -family gpt-oss
./monolith infer remote models -family llama -json
./monolith infer remote generate -model openai/gpt-oss-20b \
  -prompt 'Write one short sentence about cats.' -max-tokens 32 -stream
./monolith infer remote chat -model meta-llama/Llama-3.1-8B-Instruct
./monolith infer remote generate -model google/gemma-3-4b-it \
  -messages messages.json -json
```

The default endpoint is `https://router.huggingface.co/v1`. Model/provider
availability changes; `infer remote models` queries the current listing.
Optional HF provider suffixes such as `:fastest` or `:provider-name` pass through
unchanged. A supported client protocol does not guarantee model access.
The remote path delegates architecture and tokenization to the service, so it
can use gpt-oss, Llama, Gemma, Mistral and other models that the chosen service
actually hosts. Text-only system/user/assistant messages are supported; tools
and multimodal input are not implemented.

Mistral and GPT-2 were absent from the public router listing during the
2026-10-24 validation. Use an **existing**, appropriately configured dedicated
HF endpoint when the router does not offer the desired model:

```sh
./monolith infer remote generate \
  -endpoint https://YOUR-ENDPOINT.endpoints.huggingface.cloud/v1 \
  -model mistralai/Mistral-7B-Instruct-v0.3 -prompt 'Hello' -max-tokens 32
# For a text-generation endpoint, including a GPT-2 deployment:
./monolith infer remote generate -api text \
  -endpoint https://YOUR-ENDPOINT.endpoints.huggingface.cloud \
  -model openai-community/gpt2 -prompt 'Once upon a time' -max-tokens 32
```

This client does not provision endpoints. HTTPS endpoints are restricted to
HF's router and dedicated `*.endpoints.huggingface.cloud` hosts. Generation
redirects are rejected. Raw text mode accepts one prompt and non-streamed output;
chat mode supports interactive turns, JSON or SSE output. `-prompt-file -`
reads stdin; `-messages -` reads a JSON message array. Do not combine these input
modes. The remote JSON result preserves provider model/usage/finish metadata
when available and does not invent local token IDs.

Requests have a default two-minute timeout (`-timeout`, up to 15 minutes),
1..4096 output tokens, a 1 MiB input bound, 8 MiB response bound and 256 KiB SSE
event bound. Ctrl+C propagates cancellation and closes response bodies.
Cancellation also handles stalled stdout/stderr. The request deadline covers
stream delivery and final output; interactive input waits have no turn deadline.
Generation is not retried or switched to another backend on failure.
Truncated streams are errors, even if partial text was already printed.
Interactive history commits a reply only after successful completion. Use
`/reset` to clear it and `/exit` to quit.

## Capability and validation

| Family | Local Go execution | Remote HF execution |
| --- | --- | --- |
| GPT-2 | Tied GPT-2, raw text, pinned base model tested | Dedicated text endpoint when absent from the router |
| Llama / SmolLM2 | Restricted Llama profile; SmolLM2 135M/360M tested | Chat provider or dedicated endpoint |
| Qwen2 / Qwen2.5 | Restricted Qwen2 profile; Qwen2.5 0.5B tested | Chat provider or dedicated endpoint |
| gpt-oss | No native MoE kernel | Chat provider or dedicated endpoint |
| Mistral | Not implemented locally | Chat provider or dedicated endpoint |
| Gemma | Not implemented locally | Chat provider or dedicated endpoint |

Local Llama compatibility does not cover all Meta Llama models: size, RoPE,
tokenizer and attention restrictions still apply. Remote entries describe the
client protocol; use the live listing or your endpoint's configuration to check
the actual model.

`monolith infer models` is an offline family guide, separate from the live
hosted model listing. It describes local limitations without treating remote
protocol support as native kernel support.

Network tests use local HTTP fixtures for cache hashes, offline reuse, budgets,
atomic concurrent publication, credential redirect boundaries, rate limits,
malformed/truncated streams, cancellation, broken consumers and raw text
endpoints. A live pinned SmolLM2-135M download and offline replay produced
identical greedy output (see [receipt](validation/hf/sources-and-routing.json)).
Remote provider listing was checked live. Remote generation was **not** run
against a paid service; protocol fixtures do not establish provider quality.

Protocol references: [HF chat completion](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion),
[HF text generation](https://huggingface.co/docs/inference-providers/en/tasks/text-generation),
and [HF Hub API](https://huggingface.co/docs/hub/en/api).
