# Mistral inference

The Go importer accepts `MistralForCausalLM` with explicit
`sliding_window: null`, SiLU, bias-free projections, GQA and ordinary half-split
RoPE. This is the computational profile used by
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71).
Sliding-window models, including the original v0.1 profile, and Mixtral MoE
are not supported. A missing `sliding_window` is rejected because Transformers
otherwise defaults Mistral to a sliding window. Unsupported settings are never
silently treated as full causal attention.

This first stage supports inspection and already-framed token IDs from existing
Safetensors files. Mistral's tokenizer, automatic `infer local -repo` fetching,
text generation and chat are not yet enabled: the fetcher checks tokenizer
compatibility before downloading weights. The generic HF remote route remains
available subject to provider/endpoint availability.

```sh
./scripts/go.sh run . inspect-hf -hf-model PATH_TO_MISTRAL \
  -context 128 -weight-format q4-mixed -weight-mib 6144
./scripts/go.sh run . infer-hf -hf-model PATH_TO_MISTRAL \
  -context 128 -weight-format q4-mixed -weight-mib 6144 \
  -ids 1,23325 -max-tokens 8 -threads 4
```

The token-ID command adds no BOS or chat framing. Supply the exact IDs required
by the model. `f32`, `q8`, `q4` and `q4-mixed` share the existing streamed loader,
weight accounting, paged KV and fixed worker execution. The native training
format and its limits remain separate.

The pinned v0.3 configuration has 7,248,023,552 parameters, including its
separate output head. Its FP32 payload is 28,992,094,208 bytes, before runtime,
KV and activation memory. An explicit weight budget rejects an oversized load
before weights are opened. A smaller context does not shrink weight storage.
Structural acceptance of its 32,768-token context is not a long-context quality
or memory validation.

## Numerical coverage

Seeded synthetic Mistral fixtures use Transformers 4.57.1, Torch 2.8.0 CPU eager
attention and Safetensors 0.6.2. F32/F16/BF16 source weights are converted into
FP32 by both reference and Go. Logits and every cached K/V row are compared
with absolute tolerance `2e-5`, including chunk sizes 1/4/16/35, a page boundary,
decode, ragged sessions and one/two/four workers. Low-bit loading is additionally
checked against the reconstructed quantized weights. These are architecture
and execution tests, not evidence of real 7B task quality or performance.

The fixtures contain only small locally generated weights. Reproduce them with
`scripts/hf_mistral_fixtures.py` in the pinned environment from
`scripts/hf_reference_requirements.txt`.
