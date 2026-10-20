# Hugging Face inference

The importer accepts a deliberately restricted `LlamaForCausalLM` profile:
tied embeddings, bias-free projections, SiLU, GQA, scalar BOS/EOS IDs, and
unscaled, non-interleaved RoPE. Unknown configuration fields that could alter
computation are rejected. This is not general Transformers compatibility.

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
- Exactly the supported Llama tensor set. Tied output weights must be omitted
  from the file, as in the target model; separate `lm_head.weight` is rejected.
- Every tensor name, shape, dtype and extent is validated before allocating
  weights. Payload extents must cover each file without holes or overlaps.
- Duplicate JSON keys, malformed UTF-8, excessive nesting, missing tensors,
  unknown tensors and non-finite weight values are rejected. Value checks occur
  when reading weights, not during metadata-only inspection.
- Open descriptors bind validation and subsequent reads to the same files.
  Do not modify files in place during a load. Safetensors is not a cryptographic
  integrity check; obtain files from a trusted pinned source and retain hashes.

This first layer provides inspection and tensor decoding. Llama numerical
inference and tokenizer/serving integration are separate changes.

Format reference: [Safetensors specification](https://github.com/huggingface/safetensors#format).
Model source: [SmolLM2-135M-Instruct, pinned revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac).
