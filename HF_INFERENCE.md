# Hugging Face inference

The importer accepts restricted `LlamaForCausalLM` and `Qwen2ForCausalLM`
profiles: SiLU, GQA, scalar BOS/EOS IDs, unscaled half-split RoPE and full
causal attention. Llama projections must be bias-free; Qwen2 includes Q/K/V
biases. Both tied and separate output embeddings are supported. Sliding-window
attention, RoPE scaling, MoE and unknown computational settings are rejected.
This is not general Transformers compatibility.

| Text model | Pinned revision | FP32 weight budget | Example context |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | 600 MiB | 512 |
| SmolLM2-360M-Instruct | `a10cc1512eabd3dde888204e902eca88bddb4951` | 1600 MiB | 512 |
| Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 2048 MiB | 512 |

HF inference accepts up to one billion parameters and 262,144 vocabulary IDs,
subject to the explicit weight budget and existing activation/KV guards. Native
training/checkpoint limits remain 250 million parameters and 65,536 IDs.
Bias and separate-head storage is included before any weights are allocated.

For Qwen, download the same file set below from `Qwen/Qwen2.5-0.5B-Instruct`
at its pinned revision into `runs/qwen25`, then run:

```sh
go run . generate -hf-model runs/qwen25 -context 512 -weight-mib 2048 \
  -chat -prompt 'Goの並行処理を一文で説明して。' \
  -temperature 0 -repeat-penalty 1 -max-tokens 64
```

## Inspect a local model

Download model files separately, pinning their revision. The Go program never
downloads files or executes model repository code.

```sh
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  config.json model.safetensors tokenizer.json tokenizer_config.json \
  special_tokens_map.json generation_config.json \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --local-dir runs/smollm2
go run . inspect-hf -hf-model runs/smollm2 -context 512 -weight-mib 600
```

`inspect-hf` validates metadata and file extents without allocating model
weights. It reports FP32 weight storage, RoPE tables, KV bytes per token, the
full-context KV size for one session, source bytes, shard count and dtypes.
`-context` can reduce the declared context; it cannot extend it.
`-weight-mib` bounds FP32 weights only, **not process RSS**. Metadata (up to
4 MiB per Safetensors header), tokenizer storage, conversion buffers, Go runtime,
inference scratch and session KV require additional memory. The existing
parameter, activation and serving KV limits also apply.

For the pinned SmolLM2-135M-Instruct configuration, 134,515,008 parameters
require 538,060,032 bytes of FP32 storage. Each cached token needs 46,080 bytes;
512 tokens need 23,592,960 bytes per session before page rounding and metadata.
Its Apache-2.0 model license is separate from this repository's license.

## Storage contract

- Single `model.safetensors` or `model.safetensors.index.json` with at most
  128 local, basename-only shard paths.
- F32, F16 and BF16, converted into FP32 through a 64 KiB read buffer.
- Exactly the selected profile's tensor set. Tied output weights must be omitted;
  untied profiles require `lm_head.weight`. Qwen2 requires all Q/K/V bias tensors.
- Every tensor name, shape, dtype and extent is validated before allocating
  weights. Payload extents must cover each file without holes or overlaps.
- Duplicate JSON keys, malformed UTF-8, excessive nesting, missing tensors,
  unknown tensors and non-finite weight values are rejected. Value checks occur
  when reading weights, not during metadata-only inspection.
- Open descriptors bind validation and subsequent reads to the same files.
  Do not modify files in place during a load. Safetensors is not a cryptographic
  integrity check; obtain files from a trusted pinned source and retain hashes.

## Token-ID inference and numerical validation

```sh
go run . infer-hf -hf-model runs/smollm2 -context 512 \
  -ids 504,3575,282,4649,314 -max-tokens 8 -prefill-chunk 16 -threads 4
```

`infer-hf` accepts already framed IDs, performs chunked prefill followed by
greedy decode, and stops at the model's EOS. It adds no BOS and masks no IDs.
The diagnostic command is bounded to 256 output tokens; `-logits` includes
each next-token vector. Imported models use Llama's half-split RoPE and FP32
frequency construction. Native models retain adjacent-pair RoPE and their
existing exact checkpoint/resume behavior. Imported models cannot be written
as native `.mglm` checkpoints.

CI uses small seeded, synthetic Llama fixtures generated with Transformers
4.57.1, Torch 2.8.0 CPU eager attention and Safetensors 0.6.2. F32/F16/BF16
weights are each loaded as FP32. Tests compare logits and every KV row across
chunk sizes 1/4/16/35, a page boundary, decode, ragged batches and worker
counts 1/2/4 with absolute tolerance `2e-5`. Tiny fixtures exercise these call
paths; they do not establish throughput scaling.

The checked-in [real-model receipt](validation/hf/smollm2-numerical.json)
compares two short prompts, eight greedy tokens each and chunk sizes 1/16.
All generated IDs matched; the largest absolute logit error was
`5.340576171875e-5`, below the predefined `0.002` bound. This establishes
those numerical cases, not general task quality or long-context accuracy.
Floating-point reductions differ from Torch, so bitwise Torch parity is
not promised.

Reproduce outside the Go runtime using an isolated Python 3.11 environment:

```sh
python -m pip install -r scripts/hf_reference_requirements.txt
python scripts/hf_reference.py --fixtures testdata/hf-llama
go build -o monolith .
python scripts/hf_reference.py --model runs/smollm2 \
  --binary ./monolith --output runs/hf-numerical.json
```

The probe verifies the pinned config and weight hashes. No pretrained model
weights are checked into this repository.

## Text generation and chat

The text layer supports the SmolLM2 tokenizer profile: `Digits` (individual
Unicode numbers), GPT-2 `ByteLevel` splitting, BPE, no normalization or post
processor, and explicitly registered special tokens. It reads `tokenizer.json`
and `tokenizer_config.json`; arbitrary tokenizer pipelines are rejected.

Qwen2.5 additionally uses NFC normalization, its ordered Split/ByteLevel regex,
added tokens outside the BPE vocabulary, and trailing model vocabulary padding.
Non-special added tokens such as `<tool_call>` remain in decoded text. Padded
IDs decode to empty text, as in HF Tokenizers. The pinned Qwen chat template is
recognized by SHA-256; only system/user/assistant messages without tools are
accepted. No Jinja or model repository code executes.

NFC uses pure Go `golang.org/x/text` v0.22.0 (Go 1.22 compatible, Unicode 15).
Inputs that would require stream-safe CGJ insertion are rejected explicitly
instead of receiving different token IDs; this affects long combining-mark
sequences. Other size/token limits still apply.

```sh
go run . tokenize-hf -hf-model runs/smollm2 -prompt 'Goと猫 🐈‍⬛ 2026'
go run . generate -hf-model runs/smollm2 -context 512 \
  -prompt 'The capital of France is' -max-tokens 8 \
  -temperature 0 -repeat-penalty 1 -json
go run . generate -hf-model runs/smollm2 -context 512 \
  -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 32 -temperature 0 -repeat-penalty 1 -json
```

Plain prompts receive no implicit BOS. Empty encoded prompts are rejected for
generation. Chat formatting adds the exact supported SmolLM2 template,
including its default system message when no system message is supplied, and
the assistant generation prefix. Only the recognized SmolLM2 and Qwen2.5 templates are implemented;
Jinja is never executed. A different template remains usable for plain prompts
but fails explicitly when chat formatting is requested.

Use `-messages path/to/messages.json` instead of `-prompt/-chat` for a JSON array
of `{ "role": "user", "content": "..." }` objects. Roles are `system`, `user` and
`assistant`; at most 128 messages are accepted. `tokenize-hf` accepts the same
chat flags and reports the formatted prompt, IDs and decoded text.

Generation uses the model's scalar EOS and the existing Go sampling flags;
`generation_config.json` sampling settings are not applied. ID 256 is an
ordinary HF vocabulary ID. Generated text skips registered special tokens.
The HF decoder matches ByteLevel's lossy UTF-8 replacement behavior, including
across streamed token boundaries. Native tokenization and native UTF-8 behavior
are unchanged.

Vocabulary tables are immutable after loading and shared across requests.
Each encoding owns a priority queue of BPE merge candidates and checks context
cancellation. Prompts are limited to 1 MiB, individual regex pre-tokens to
64 KiB, tokenizer JSON to 32 MiB and expanded vocabulary pieces to 16 MiB.
Context overflow is rejected without truncation. Unicode categories follow
Go's Unicode tables. Some byte-alphabet entries are absent from SmolLM2's
vocabulary; with its null BPE unknown token, those bytes are omitted exactly as
in Hugging Face Tokenizers. Thus this tokenizer is not byte-lossless for every
input, unlike the native tokenizer.

## Serve the imported model

```sh
go run . serve -hf-model runs/smollm2 -context 512 -weight-mib 600 \
  -threads 4 -kernel-workers 2 -max-batch 2 -queue 4 -kv-mib 64

curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write one short sentence about cats."}],"max_tokens":32,"temperature":0,"repeat_penalty":1,"stream":true}'
```

The existing project-specific `/v1/completions` accepts either `prompt` or
`messages`, with the same completion JSON and SSE token/done/error events.
`messages` requires supported HF chat formatting; the two input forms cannot
be combined. This endpoint is not an OpenAI Chat Completions implementation.
`-model` and `-hf-model` are mutually exclusive.

Imported models use the same actor, fixed per-engine workers, bounded admission,
KV page pool, cancellation, slow-consumer handling and shutdown lifecycle as
native models. No Python, cgo, external server or ML runtime participates in
Go inference.

The [text-serving receipt](validation/hf/smollm2-serving.json) records matching
CLI, HTTP and two concurrent SSE outputs against Transformers for a raw prompt
and a 37-token chat prompt, with eight generated tokens each. All four HTTP
requests released admission and KV reservations; retained allocated pages are
reusable pool storage. Server shutdown completed normally. This small probe is
a compatibility/lifecycle check, not a production load benchmark.

The [tokenizer reference](validation/hf/smollm2-tokenizer.json) contains
147 encoding cases, three chat cases and six decoding cases, including numbers,
whitespace, contractions, Japanese, emoji, combining marks, special tokens and
missing byte entries. All matched Go locally. CI runs equivalent tests on a
small synthetic tokenizer fixture without downloading a model.

```sh
python scripts/hf_tokenizer_reference.py --fixture testdata/hf-tokenizer
python scripts/hf_tokenizer_reference.py --model runs/smollm2 \
  --output runs/tokenizer-reference.json
MONOLITH_HF_MODEL=runs/smollm2 \
MONOLITH_HF_TOKENIZER_REFERENCE=runs/tokenizer-reference.json \
  go test -race -run TestHFLocalTokenizerReference -v .
python scripts/hf_serving_reference.py --model runs/smollm2 \
  --binary ./monolith --output runs/hf-serving.json
```

Format reference: [Safetensors specification](https://github.com/huggingface/safetensors#format).
Model source: [SmolLM2-135M-Instruct, pinned revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac).

## Additional profile validation

Seeded fixtures cover tied Qwen2 F32/F16, untied Qwen2 BF16 and untied Llama
F32. Qwen biases are deliberately nonzero. Tests compare every logit and KV
row across worker counts, chunk sizes, a page boundary and ragged sessions
with absolute tolerance `2e-5`. A synthetic Qwen tokenizer exercises NFC,
regex boundaries, ordinary added tokens, special tokens and padding IDs.

The [model manifest](validation/hf/model-profiles.json) binds files by SHA-256;
weight hashes are checked against the Hub's LFS metadata. Real-model probes
use two short prompts and eight greedy tokens per prompt, chunk sizes 1/16,
CLI, HTTP and two concurrent SSE requests. These are bounded compatibility
and lifecycle checks, not model-quality or production-load claims.

```sh
python scripts/hf_profile_fixtures.py
python scripts/hf_qwen_tokenizer_fixture.py --source runs/qwen25
python scripts/hf_model_reference.py --id qwen25 --model runs/qwen25 \
  --binary ./monolith --output runs/qwen25-reference.json
```

Use `--id smollm2-360m` for the pinned 360M model. These offline scripts never
fetch models. The oracle requires `scripts/hf_reference_requirements.txt`.

The [Qwen2.5 receipt](validation/hf/qwen25-serving.json) records a maximum
absolute logit error of `8.296966552734375e-5` (limit `0.002`) and matching CLI,
HTTP and concurrent SSE output. Its [tokenizer oracle](validation/hf/qwen25-tokenizer.json)
covers 159 encodings, three chats and six decode cases.

The [SmolLM2-360M receipt](validation/hf/smollm2-360m-serving.json) records
maximum absolute logit error `5.054473876953125e-05` (limit `0.002`), matching text on
all three interfaces, and released request resources. Its
[tokenizer oracle](validation/hf/smollm2-360m-tokenizer.json) covers the same 159/3/6 cases.
