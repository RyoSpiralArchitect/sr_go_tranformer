# Mistral inference

The local Go path supports the full-attention `MistralForCausalLM` profile and
the v3 Metaspace/ByteFallback tokenizer used by
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71).
It provides raw text, plain system/user/assistant chat, HTTP/SSE, automatic HF
caching and `HF_HUB_OFFLINE=1` reuse. Native Go training remains a separate path.

Numerical tests use small synthetic models. The official tokenizer has been
checked locally, but the complete 7B weights could not be obtained during this
validation run. Real 7B generation, quality, throughput and peak RSS are therefore
**not yet validated**. See [the recorded boundary](validation/hf/mistral-v03-validation.json).

## Run locally

This downloads about 14.5 GB of original BF16 Safetensors on the first run.
`q4-mixed` reduces resident weight storage, not download size. Set a small context
first; the budgets below are limits, not measured total process memory.

Prepare the verified snapshot separately from inference:

```sh
./scripts/go.sh run . infer local \
  -repo mistralai/Mistral-7B-Instruct-v0.3 \
  -revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  -cache-dir runs/hf-cache -download-mib 15000 -download-timeout 2h \
  -context 128 -weight-format q4-mixed -weight-mib 6144 fetch -json
```

Network failures and Ctrl-C retain progress; rerun the same command to resume.
`fetch` verifies source hashes and model metadata without loading weight tensors.
Only a complete snapshot is usable offline. See [download and resume behavior](INFERENCE.md)
for concurrent downloads, deadlines and abrupt-termination limitations.

```sh
./scripts/go.sh run . infer local \
  -repo mistralai/Mistral-7B-Instruct-v0.3 \
  -revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  -cache-dir runs/hf-cache -download-mib 15000 \
  -context 128 -weight-format q4-mixed -weight-mib 6144 \
  generate -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 16 -temperature 0 -threads 4
```

After a complete snapshot has been published to that cache, the same request
can run without network access:

```sh
HF_HUB_OFFLINE=1 ./scripts/go.sh run . infer local \
  -repo mistralai/Mistral-7B-Instruct-v0.3 \
  -revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  -cache-dir runs/hf-cache -context 128 \
  -weight-format q4-mixed -weight-mib 6144 \
  generate -chat -prompt 'Hello!' -max-tokens 16 -threads 4
```

Omit `-chat` for raw text; use the `chat` action for interactive history/reset.
Existing local Safetensors exports use `-dir PATH_TO_MISTRAL` in place of the
repository/cache/download flags. `serve` exposes the same completion and chat
input through `/v1/completions`; see [HTTP serving](SERVING_VALIDATION.md). To keep weights
off the host entirely, choose [HF remote inference](INFERENCE.md) explicitly;
provider availability is separate from local model support.

## Compatibility and resources

- Require explicit `sliding_window: null`, SiLU, bias-free GQA projections and
  ordinary half-split RoPE. Missing or enabled sliding windows are rejected.
  The original v0.1 sliding-attention profile and Mixtral MoE are unsupported.
- Recognize the exact v3 tokenizer pipeline and pinned HF chat template.
  Tools/function calls and arbitrary Jinja templates are unsupported. An
  unrecognized chat template disables chat while leaving valid raw text usable.
- Raw text adds one BOS even if the input begins with literal `<s>`. Formatted
  chat supplies its own BOS. Chat roles alternate user/assistant after an
  optional initial system message. The official template inserts that system
  message at the final user turn; it is not moved to the first turn.
- Special tokens are skipped in decoded output. Metaspace removes one leading
  space. ByteFallback buffers a contiguous byte-token run until its boundary,
  preserving the official invalid-UTF-8 behavior in both completion and SSE.
  Some token events can therefore have empty text. Buffering is bounded by the
  request's generation/context budget.
- Input must be valid UTF-8 and at most 1 MiB; each transformed ordinary segment
  between added tokens is capped at 64 KiB. Token/context budgets still apply.
- `f32`, `q8`, `q4` and `q4-mixed` share the streamed loader, paged KV and fixed
  workers. The pinned v0.3 configuration has 7,248,023,552 parameters, including
  its separate output head. FP32 payload alone is 28,992,094,208 bytes. Smaller
  contexts do not shrink weights. Structural acceptance of 32,768 positions
  does not establish long-context quality or memory use.

The Go fetcher validates metadata and tokenizer compatibility before requesting
weights, verifies file identities and publishes complete snapshots atomically.
An offline cache miss or unsupported profile fails explicitly.

## Validation and reproduction

Seeded synthetic Mistral fixtures use Transformers 4.57.1 / Torch 2.8.0 CPU eager
attention / Safetensors 0.6.2. F32/F16/BF16 sources are converted to FP32 by both
implementations. Tests compare logits and every K/V row at absolute tolerance
`2e-5`, including chunks 1/4/16/35, page boundaries, decode, ragged sessions and
one/two/four workers. Low-bit tests compare against reconstructed quantized
weights. Generate them with `scripts/hf_mistral_fixtures.py`.

The synthetic tokenizer covers 174 text, 7 chat and 12 decoding cases, including
Unicode, special-token boundaries, whitespace and invalid byte runs. The same
cases pass against the pinned official tokenizer. Default Transformers
`AutoTokenizer` conversion is also compared with the serialized tokenizer.
CLI, HTTP/SSE, EOS, resource release and an offline cache round trip have small
fixture tests; these do not load pretrained 7B weights.

```sh
# Use scripts/hf_reference_requirements.txt in an isolated Python environment.
python -I scripts/hf_mistral_tokenizer_fixture.py \
  --source PATH_TO_PINNED_METADATA --output testdata/hf-mistral-tokenizer
python -I scripts/hf_mistral_tokenizer_fixture.py \
  --source PATH_TO_PINNED_METADATA --output runs/mistral-tokenizer.json --real
MONOLITH_HF_MODEL=PATH_TO_PINNED_METADATA \
MONOLITH_HF_TOKENIZER_REFERENCE=runs/mistral-tokenizer.json \
  ./scripts/go.sh test -run TestHFLocalTokenizerReference -v .
```

The real oracle additionally needs the official `tokenizer.model`, whose hash
is recorded in `validation/hf/model-profiles.json`. The Go runtime only needs
`tokenizer.json` and `tokenizer_config.json`; it does not depend on Python or
SentencePiece. Once all pinned weights are available, run
`scripts/hf_large_model_smoke.py --id mistral-7b-v03` for bounded real CLI,
HTTP, concurrent SSE, cancellation and resource-release checks.
