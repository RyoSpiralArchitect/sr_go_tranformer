# Llama 3 text inference and a 3B numerical trial

`inspect-hf` and `infer-hf` accept the bounded Llama 3.1/3.2 RoPE scaling
profile (`rope_type: llama3`) and an optional `head_dim` equal to
`hidden_size / num_attention_heads`. This enables the pinned, locally cached
Llama 3.2 3B BF16 export to run directly in Go, including Q8, Q4 and mixed Q4/Q8.
The original model configuration and weight files are used unchanged.

The pinned Llama 3.2 base and Instruct tokenizers support `tokenize-hf`, plain-text
`generate`, HTTP completion/SSE and the `infer local` download path. Its exact
Split/ByteLevel BPE profile uses 1–3 Unicode-number groups, no normalization,
whole-pre-token vocabulary lookup (`ignore_merges`), and BOS post-processing.
Plain text receives one BOS even when the supplied text already contains a
literal BOS. `infer-hf` continues to take already framed IDs without adding BOS.

Decoding skips registered special tokens and follows the tokenizer's configured
Transformers whitespace cleanup. Streaming uses bounded pending prefixes, so
concatenated text deltas equal the final completion, including punctuation,
contractions, trailing spaces and split UTF-8. Unsupported pipelines still fail.
The base model has no chat template and continues to reject chat. The recognized
Instruct template supports text-only system/user/assistant messages through CLI,
interactive chat, HTTP and SSE. This does not turn a base model into an
instruction-following one.

The original numerical experiment below prepared IDs with pinned Tokenizers;
all model loading, matrix computation, KV storage and generation ran in Go.
The new text path is independently checked against that same pinned tokenizer.
The [real tokenizer oracle](validation/llama3/text-tokenizer.json) has 173
encoding and nine decoding cases; all match, including bytewise streaming.
The [text CLI receipt](validation/llama3/text-cli.json) reproduces the first
eight-token continuation from the earlier token-ID trial.
The Llama Split letter, number and whitespace classes are pinned to Tokenizers
0.22.2 rather than the Go toolchain's Unicode release. Their ranges are generated
over every Unicode scalar by `scripts/hf_llama3_unicode.py`; regression inputs
cover characters whose classification changes across recent Unicode releases.

## Use existing model files

The measured source is `mlx-community/Llama-3.2-3B-bf16`, revision
`60a99aaf43164077157d64bf909b7b61143c6a6d`: two BF16 Safetensors shards totaling
6,425,528,971 bytes. It is a **base model**, not an instruction-tuned model.
The export contains ordinary HF tensor names; MLX is not used to execute it.
[Source hashes](validation/hf/model-profiles.json) identify every input file.
The model's Llama license is separate from this repository's Apache-2.0 license;
pretrained weights are not redistributed here.

```sh
CGO_ENABLED=0 go build -trimpath -o monolith .
./monolith inspect-hf -hf-model PATH_TO_LLAMA32_3B -context 128 \
  -weight-format q4-mixed -weight-mib 2000
# Already-framed IDs for "The capital of France is", including BOS 128000.
HF_HUB_OFFLINE=1 ./monolith infer-hf -hf-model PATH_TO_LLAMA32_3B \
  -context 128 -weight-format q4-mixed -weight-mib 2000 \
  -ids 128000,791,6864,315,9822,374 -max-tokens 8 \
  -prefill-chunk 16 -threads 4
HF_HUB_OFFLINE=1 ./monolith generate -hf-model PATH_TO_LLAMA32_3B \
  -context 128 -weight-format q4-mixed -weight-mib 2000 \
  -prompt 'The capital of France is' -max-tokens 8 \
  -temperature 0 -repeat-penalty 1 -json
```

`infer-hf` neither tokenizes text nor adds BOS. It emits generated token IDs;
use the matching tokenizer to decode them. `check-hf` validates the Go text
profile; `check-hf -chat` correctly rejects this base model.

The model has 3,212,749,824 parameters. Mixed weights use 2,004,774,912 bytes,
including scales and FP32 norms, versus 12,850,999,296 bytes for FP32. A context
of 128 requires 29,360,128 KV payload bytes per full session, before page rounding
and metadata. `-weight-mib` limits weight payload only; runtime, allocator,
metadata, workspace and KV are additional. Loading streams through bounded
buffers without a second full FP32 weight copy.

## Instruct chat

The pinned 1B Instruct source is
`mlx-community/Llama-3.2-1B-Instruct-bf16@863c846a9ac6fad4e49e1743d52984dff262e953`.
Its original BF16 files use the same Llama computational profile; MLX is not
required. The model license is Llama 3.2, separately from this project's license.

```sh
./monolith infer local -repo mlx-community/Llama-3.2-1B-Instruct-bf16 \
  -revision 863c846a9ac6fad4e49e1743d52984dff262e953 \
  -download-mib 2500 -weight-format q8 -weight-mib 1400 -context 512 \
  chat -max-tokens 64

HF_HUB_OFFLINE=1 ./monolith generate -hf-model PATH_TO_LLAMA32_INSTRUCT \
  -weight-format q8 -weight-mib 1400 -context 512 \
  -chat -prompt 'Reply with exactly OK.' -max-tokens 8 \
  -temperature 0 -repeat-penalty 1 -json
```

`-messages FILE` / `-messages -` and HTTP `messages` use the same formatter.
The exact template is identified by SHA-256; arbitrary Jinja, tools, tool calls,
multimodal input and unknown roles are rejected. It adds the template's system
header, knowledge date, current local calendar date, trimmed message content,
end-of-turn markers and assistant header. The local clock is read for each chat
prompt. Golden tests provide a fixed date; real receipts preserve the rendered
prompt so that dynamic date text is visible.

Chat supplies its own BOS and explicitly bypasses plain-text BOS processing.
A literal BOS in ordinary text is still preserved separately.
Llama config `eos_token_id` may be one integer or 1–8 distinct in-range integers.
Every ID must be a registered special token for text inference. All configured
stop IDs terminate token-ID, CLI, HTTP and SSE generation with `finish_reason`
set to `eos`; the terminating ID is retained in accounting but hidden in text.
Inspection exposes the stop set. No extra stop IDs are invented from names.

The fixed-date tokenizer oracle covers seven chat arrangements, including
system-only input, history, Unicode trimming and empty content. Synthetic
generation tests force every configured termination ID and check one BOS,
stream completion, HTTP equivalence and resource release. CLI subprocess tests
also compare direct flags, stdin text, stdin message JSON and interactive reset.
The pinned 3B Instruct metadata at revision
`6d88ba43024fef71b10e52e101c7cd4598322601` has identical tokenizer and template
files to the 1B profile. Its 159 encoding, seven chat and nine decoding cases
match the same oracle; this metadata check does not execute the 3B Instruct
weights. [Tokenizer verification](validation/llama3/chat-tokenizer-validation.json)
records both sources. Base models and
unrecognized templates retain only their supported plain-text paths.

The [1B Instruct real-model receipt](validation/llama3/chat-model-reference.json)
compares Go FP32 with Transformers 4.57.1 / Torch 2.8.0 CPU eager attention.
For two bounded requests, generated IDs and final text match at prefill chunks
1 and 16; maximum absolute logit error is `0.000129313` (tolerance `0.002`).
`Reply with exactly OK.` produces `[4012, 128009]`, text `OK`, and an EOS finish.
The ordinary-text request continues ` Paris. The Eiffel Tower is` for eight
tokens. CLI, HTTP and concurrent SSE agree, all four serving requests release
their reservations, and shutdown is clean. This validates these entry paths and
numerical cases, not general instruction-following quality. The 3B Instruct
profile above is checked at the tokenizer level only.
The [CLI smoke receipt](validation/llama3/chat-cli-smoke.json) separately checks
stdin text/message input, interactive reset and an idle-stdin SIGINT exit using
the same real weights and expected completion.

```sh
# Matching pinned 1B Instruct weights and the isolated reference environment.
HF_HUB_OFFLINE=1 python scripts/hf_model_reference.py \
  --model PATH_TO_LLAMA32_1B_INSTRUCT --id llama32-1b-instruct \
  --binary ./monolith --output runs/llama1b-reference.json
# The same published tokenizer oracle also checks pinned 3B Instruct metadata.
MONOLITH_HF_MODEL=PATH_TO_LLAMA32_3B_INSTRUCT \
  MONOLITH_HF_TOKENIZER_REFERENCE=validation/llama3/chat-tokenizer.json \
  go test -run '^TestHFLocalTokenizerReference$' -count=1 .
```

The FP32 reference run uses a 5,100 MiB weight budget and releases the Torch
model before loading Go weights; allow extra RAM for tokenizer, runtime and KV.
The Q8 command above has a smaller weight budget and does not promise FP32
numerical equality.

## RoPE contract

The fixed Llama 3 algorithm preserves high frequencies, divides low frequencies
by `factor`, and interpolates the middle region using the original context and
low/high frequency factors. Frequency construction, scaling and position
products use FP32 operation order. JSON scalars retain FP64 precision through
scalar-only differences/divisions, then convert at the corresponding tensor
operation; rounding thresholds earlier can change interpolation materially.
The implementation is checked against
[Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py).
It applies at every position, including short prompts.

The loader accepts only the five required fields: `rope_type`, `factor`,
`low_freq_factor`, `high_freq_factor`, and `original_max_position_embeddings`.
Unknown fields and scaling algorithms fail. Factor must be finite in [1, 1e6],
low frequency factor at least 1e-6, high factor greater than low and at most 1e6,
and original context an integer in [1, 131072] within the model's declared
context. Exact case-sensitive keys are required; aliases such as `FACTOR`
cannot override `factor`. Validation also checks the FP32 thresholds so equal rounded thresholds
cannot produce division by zero. Qwen2 scaling remains unsupported.

A runtime context cap does not change pretrained scaling parameters. The
runtime's existing 32,768-token context ceiling and allocation guards remain;
a model declaring 131,072 positions does not enable 128K execution. Native
training/checkpoint behavior is unchanged. Inspection reports `rope_scaling`
alongside the selected weight layout.

## Independent reference and real-model scope

Three tiny, seeded F32/F16/BF16 fixtures cover all three frequency regions,
positions crossing the original context and a KV page, GQA, prefill/decode,
ragged batches, borrowed workspaces, and one/two/four workers. Logits and every
KV row match the independent Transformers CPU eager-attention oracle within
`2e-5` absolute error. A separate inverse-frequency oracle covers the real 3B
configuration and fractional thresholds, including a narrow interpolation band.
Low-bit fixtures compare against reconstructed quantized
weights; they do not impose FP32 equality.

```sh
# Install scripts/hf_reference_requirements.txt in an isolated reference venv.
HF_HUB_OFFLINE=1 python scripts/hf_reference.py \
  --fixtures testdata/hf-llama3 --llama3
go test -run 'Test(HFLlamaTransformersReference|Llama3ConfigBoundaries|QualityExternalTokenBoundaries)' .
```

The real 3B trial reuses the six original passages and 96 teacher-forced positions
from [the quality probe](QUALITY_INFERENCE.md), with separately prepared input
IDs. No new model download is needed for an existing verified snapshot. The
input file binds the exact corpus, model revision, tokenizer hash/version and
IDs. Go checks lengths, vocabulary bounds and identity before scoring; the
reference-vector header binds the actual scored IDs. This separates tokenizer
preparation from the imported model's numerical behavior.

```sh
# Tokenizers 0.22.2 from the isolated reference environment.
python scripts/quality_tokens.py --model PATH_TO_LLAMA32_3B --id llama32-3b \
  --output runs/llama3-tokens.json
python3 -S scripts/quant_quality.py --model PATH_TO_LLAMA32_3B --id llama32-3b \
  --token-inputs runs/llama3-tokens.json --weight-mib 13000 \
  --output-dir runs/llama3-quality
python scripts/quality_tokens.py --model PATH_TO_LLAMA32_3B --id llama32-3b \
  --receipt runs/llama3-quality/receipt.json --output runs/llama3-text.json
```

Each format runs in a separate process. The FP32 trial needs physical memory
beyond its roughly 12 GiB weight payload; the measured host was an Apple M4 with
24 GiB RAM. With less memory, use `--reference q8 --formats q8 q4 q4-mixed` and
label comparisons as Q8-relative. The runner defaults to an 8 GiB weight budget;
the command above explicitly raises it to 13,000 MiB (the runner caps it at
16,384 MiB). The production HF parameter ceiling stays at eight billion.

The full 3B model is compared to its Go FP32 path; independent Transformers
agreement is established by the tiny fixtures, not by a full-model Torch run.
These short English passages and continuations do not establish general
language quality, instruction following or long-context accuracy. External
text rendering uses backend Tokenizers decoding without Transformers whitespace
cleanup; Go's raw generated IDs are retained unchanged.

## Measured 3B results

On Apple M4 / 24 GiB / Go 1.27.1, the 96-position probe produced:

| Format | Weight payload, bytes | Mean NLL | KL from FP32 | Top-1 matches / 96 |
| --- | ---: | ---: | ---: | ---: |
| f32 | 12,850,999,296 | 3.0772 | 0 | 96 |
| q8 | 3,414,061,056 | 3.0817 | 0.000302 | 93 |
| q4 | 1,807,773,696 | 3.1067 | 0.119411 | 74 |
| q4-mixed | 2,004,774,912 | 3.0262 | 0.063014 | 82 |

Mixed storage uses 84.4% less weight payload than FP32 and 41.3% less than Q8,
while using 10.9% more than Q4. It reduced distribution drift relative to Q4 on
this probe. Its slightly lower NLL than FP32 is a result on this small sample,
not evidence of a generally better model. Q8 remained closest to FP32.

One post-probe RSS observation per process was 12.04 GiB (FP32), 3.27 GiB (Q8),
1.77 GiB (Q4) and 1.96 GiB (mixed). These are not peak RSS measurements or an
RSS guarantee; timing fields include validation/I/O and are not a throughput
comparison. Model formats ran sequentially with context 128 and four Go kernel
workers; the scoring batch had six documents.

For `The capital of France is`, all formats continued with
` Paris. It is the largest city in`. For `Once upon a time, a small cat`,
FP32/Q8/mixed continued ` named Tigger was born in a forest`, while Q4 produced
` named Puss in Boots was born.` Both samples' eight generated IDs matched FP32
in the mixed path; that does not imply equality on other prompts or positions.
A separate public `infer-hf` CLI run reproduced the mixed first continuation.

[Raw quality receipt](validation/llama3/quality.json),
[framed input IDs](validation/llama3/inputs.json),
[external text rendering](validation/llama3/text.json),
[CLI receipt](validation/llama3/cli.json), and
[environment](validation/llama3/environment.json) retain hashes, per-position
measurements and raw generated IDs. The 7B model was not evaluated in this run;
its source transfer was deferred when the connection remained too slow.
