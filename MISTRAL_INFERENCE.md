# Mistral inference

The local Go path supports the full-attention `MistralForCausalLM` profile and
the v3 Metaspace/ByteFallback tokenizer used by
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71).
It provides raw text, plain system/user/assistant chat, HTTP/SSE, automatic HF
caching and `HF_HUB_OFFLINE=1` reuse. Native Go training remains a separate path.

Numerical parity tests use small synthetic models and an official tokenizer
reference. The complete pinned 7B files have now also been verified locally and
used for bounded Q8/Q4-mixed generation and memory probes. Q4-mixed CLI, HTTP
and concurrent SSE agree, with cancellation and resource release checked.
These short probes do not establish general language quality. A separate
[long-generation comparison](LONG_GENERATION.md) measures a 256-token prefix
and 256 output IDs with exact numerical checks. See [local validation](validation/hf/mistral-v03-local-20261029.json)
and the [earlier incomplete-download record](validation/hf/mistral-v03-validation.json).

## Run locally

If the model already exists on disk, point directly at its folder or index.
This reads the original files in place without copying or downloading them:

```sh
HF_HUB_OFFLINE=1 ./scripts/go.sh run . infer local \
  -path "$HOME/models/mistral7b-instruct-v0.3" \
  -context 128 -weight-format q4-mixed -weight-mib 6144 \
  generate -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 16 -temperature 0 -threads 4
```

`-path` also accepts `config.json`, `model.safetensors.index.json` or a shard
listed by that index. All three original BF16 shards must be present. Use
`check -chat` as the action to inspect compatibility without loading weights.
These are original source weights quantized by Go during loading; prequantized
GGUF/GPTQ/AWQ/MLX formats remain unsupported.

The `-repo` examples below download about 14.5 GB of original BF16 Safetensors
on the first run. Existing `-path` files are read locally.
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
SentencePiece. To repeat the pinned local-model checks, run
`scripts/hf_large_model_smoke.py --id mistral-7b-v03` for bounded real CLI,
HTTP, concurrent SSE, cancellation and resource-release checks.

## Bounded real 7B results

On 2026-10-29, Apple M4 / 24 GiB / Go 1.27.1, all three local shards matched
the pinned source hashes. The existing quality corpus supplied three passages
and eight teacher-forced positions per passage (24 scored tokens), plus two
eight-token greedy raw-text continuations. Context was 128 with four workers;
each format ran in a separate process, one resident model at a time.

| Format | Weight payload, GiB | Process peak RSS, GiB | Mean KL from Q8 | Next-token matches with Q8 |
| --- | ---: | ---: | ---: | ---: |
| Q8 | 7.173 | 7.108 | 0 | 24/24 |
| Q4-mixed | 3.923 | 4.025 | 0.03043 | 20/24 |

The mixed payload is 45.3% smaller. Both short greedy continuations happened
to match, despite the distribution and argmax differences. Q8 is the reference
here, not FP32 or Transformers; FP32 payload alone exceeds this host's RAM.
The 24-token NLL values (3.88580 / 3.87494) are diagnostic, not a ranking of
general quality. Peak RSS uses each child's `wait4` lifetime maximum, including
load and probing. Host activity and file caches were uncontrolled; reported
timings are not an optimization comparison.

[Quality receipt](validation/quality/mistral7b-v03-q8-q4mixed-20261029.json) and
[serving receipt](validation/hf/mistral-v03-local-serving-20261029.json) retain
source/build identities, outcomes and measurements. The serving trial uses
`infer local -path` with `HF_HUB_OFFLINE=1`, Q4-mixed, raw text and chat, two
concurrent streams, one canceled request, drained reservations and clean shutdown.
