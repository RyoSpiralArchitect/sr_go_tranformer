# Q8 local inference

`-weight-format q8` stores supported HF model matrices in signed 8-bit blocks
and computes with FP32 activations. It is opt-in; `f32` remains the default.
The same local CLI, chat and serving engine work with either format.

```sh
go run . infer local -repo openai-community/gpt2 \
  -revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  -context 128 -weight-format q8 -weight-mib 140 check -json
go run . infer local -repo openai-community/gpt2 \
  -revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  -context 128 -weight-format q8 -weight-mib 140 -offline \
  generate -prompt 'The capital of France is' -max-tokens 8 \
  -temperature 0 -repeat-penalty 1 -json
```

For direct commands, the same flag is accepted by `check-hf`, `inspect-hf`,
`infer-hf`, `generate`, `chat` and `serve` with `-hf-model`. Native checkpoints
and training reject the HF weight-format flag.

## Storage and execution

Every output row is split into blocks of at most 64 weights. Each block stores
one FP32 scale and signed integers in [-127, 127]. The scale is approximately
`max(abs(weight)) / 127`; values round to the nearest integer with ties away
from zero. Zero blocks have zero scales; subnormal and extreme finite inputs
are handled explicitly. A full block uses 68 bytes instead of 256 bytes, about
73.4% less weight payload. Partial rows require their own final scale.

Token embeddings, tied or separate output heads, and attention/MLP matrices
use Q8. Normalization weights, biases and GPT-2 position embeddings remain
FP32. Activations, accumulators, RoPE and session KV also remain FP32. This is
an application-specific weight-only format, not a GGUF, GPTQ, AWQ or LLM.int8()
importer. Prequantized source files remain unsupported.

The loader allocates the selected storage directly. Row-major Safetensors
use a 64 KiB byte reader plus one 256-byte FP32 block. GPT-2 Conv1D matrices
need two streamed reads to find scales and transpose/split QKV into their
destinations. Neither path materializes a full FP32 tensor before quantizing.
Source files must remain unchanged throughout a load.

The kernel consumes the integer weights directly, accumulating four independent
partial sums per row to expose instruction-level parallelism. Four-row tiles
reuse each conversion across prompt/session rows. Full and partial tiles use
the same per-row reduction order, so changing batch shape or worker count
does not change Q8 results on the same platform. Fixed Go workers share
immutable weights; each engine owns its bounded workspace and session KV.
No cgo, assembly, GPU runtime or external inference library is required.

Quantization occurs on every load. GPT-2's second projection read and scale
calculation can increase startup time. HF cache files retain their original
dtype and size; Q8 reduces resident weights, not download size or disk usage.
`HF_HUB_OFFLINE=1` works with verified cached snapshots in either format.

## Memory planning and larger models

`check` and `inspect` report `weight_format`, `weight_bytes`,
`fp32_weight_bytes`, quantized parameter count, scale bytes and conversion
buffers. `-weight-mib` bounds the complete selected weight payload before
weight allocation or download. It excludes allocator overhead, metadata,
tokenizer, runtime, scratch, RoPE and KV: it is not a process memory limit.
Context and `serve -kv-mib` remain separate controls.

The HF parameter ceiling is 8 billion; dimensions, tokenizer/architecture
profiles and activation/KV limits still apply. This permits larger compatible
models without relaxing the 250-million-parameter native training limit.
It does not make arbitrary Llama, Gemma, Mistral or MoE models compatible.

For a pinned Qwen2.5 1.5B model (about 3.09 GB of source weights):

```sh
go run . infer local -repo Qwen/Qwen2.5-1.5B-Instruct \
  -revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  -download-mib 4096 -weight-mib 1700 -weight-format q8 -context 128 \
  check -json
HF_HUB_OFFLINE=1 go run . infer local -repo Qwen/Qwen2.5-1.5B-Instruct \
  -revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  -weight-mib 1700 -weight-format q8 -context 128 \
  generate -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 24 -temperature 0 -repeat-penalty 1
```

Check available host memory beyond the payload estimate. Successful metadata
preflight neither loads weight values nor guarantees available physical RAM.

## Validation and reproduction

Q8 changes logits and can change generated text. The tests check the kernel
against a dot product of reconstructed weights; they do not impose FP32 token
parity. Finite extremes, odd row widths, worker/batch invariance, F32/F16/BF16
sources, separate heads, GPT-2 transpose/split, exact storage budgets, offline
source reuse, cancellation and independent engine shutdown are covered.

The real-model probe compares next-token logits at identical input prefixes
against this binary's FP32 path. Independent eight-token greedy continuations
are recorded separately, including disagreements. It also requires CLI,
HTTP and simultaneous SSE outputs to agree within each format. This small
probe is not a perplexity benchmark or a general quality assessment.

```sh
CGO_ENABLED=0 go build -trimpath -o monolith .
go build -o runs/loadtest ./tools/loadtest
python3 -S scripts/q8_reference.py --model PATH_TO_PINNED_MODEL --id gpt2 \
  --binary ./monolith --loadtest runs/loadtest --output runs/q8-gpt2.json
GOMAXPROCS=4 go test -run '^$' -bench '^BenchmarkQ8Linear$' \
  -benchmem -benchtime=250ms -count=3 -cpu=4
```

The probe uses only Python's standard library, hash-checks the source catalog,
forces offline Go execution and sends HTTP only to its own loopback server.
Its three trials alternate FP32/Q8 order. Each fresh server uses four threads
and kernel workers, context 128, maximum batch 2, queue 4, 32 MiB KV, prefill
chunk 8, mixed chunk 4 and 16 tokens per tick. Two warmup requests precede
eight measured requests at concurrency 2, with eight greedy output tokens
per request and alternating short/long prompts. Startup, correctness checks
and warmup are outside the timed throughput interval.
