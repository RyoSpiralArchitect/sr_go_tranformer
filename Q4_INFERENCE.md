# Experimental packed Q4 inference

`-weight-format q4` keeps supported HF matrices in four-bit blocks and uses a
shared 2 KiB table to decode pairs of weights. It is an opt-in memory/throughput
tradeoff. **It changes outputs substantially on some models.** FP32 is still
the default; Q8 is the more conservative quantized option.

```sh
go run . infer local -repo Qwen/Qwen2.5-1.5B-Instruct \
  -revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  -download-mib 4096 -weight-format q4 -weight-mib 900 -context 128 \
  check -json
HF_HUB_OFFLINE=1 go run . infer local -repo Qwen/Qwen2.5-1.5B-Instruct \
  -revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  -download-mib 4096 -weight-format q4 -weight-mib 900 -context 128 \
  generate -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 16 -temperature 0 -repeat-penalty 1
```

Put source flags before the action. Direct `-hf-model` commands also accept
`-weight-format q4`. The architecture profiles and eight-billion-parameter HF
ceiling are unchanged. Native checkpoints/training reject this flag. This is
not a GPTQ, AWQ, GGUF or prequantized Safetensors importer.

## Layout and Go execution

Each output row has independent blocks of at most 64 weights. A block has
one FP32 scale, approximately `max(abs(weight))/7`, and signed integers in
[-7, 7]. Rounding uses nearest integer with ties away from zero. Two values
share a byte, low nibble first. An odd-width row has one unused zero nibble;
padding never crosses a row. Subnormal and extreme finite weights are handled
explicitly. A full block uses 36 bytes, versus 68 for Q8 and 256 for FP32.

Token embeddings, tied or separate output heads and attention/MLP matrices
use Q4. Normalization weights, biases and GPT-2 position embeddings stay FP32.
The same is true of activations, ordinary accumulation, RoPE and KV storage.
`weight_bytes` includes scales, row padding and the remaining FP32 weights.

The loader writes directly into packed destinations with the same bounded
64 KiB reader and 256-byte conversion block as Q8. GPT-2 transposition and QKV
splitting use two streamed reads. There is no full floating-point duplicate
and no model-sized temporary integer buffer. Quantization runs at every load;
cached source files and download sizes do not shrink. Keep source files
immutable while loading.

The kernel shares a read-only `[256][2]float32` decode table across engines.
It consumes two packed bytes at a time and reuses four decoded values across
up to four prompt/session rows. Independent output tiles go to the existing
fixed Go workers, which join before the next layer reuses the workspace.
The table is allocated once; no per-request lookup table, decompressed matrix,
cgo, unsafe pointers, assembly or architecture-specific backend is required.
The existing callback dispatch still allocates once per kernel invocation.

Reduction order is fixed across worker counts and batch shapes. The conservative
Q8 activation threshold also selects a per-row FP64 accumulation fallback for
Q4 when unusually large values could overflow an unscaled block partial sum.
This avoids changing ordinary rows merely because a neighboring row is extreme.

`-weight-mib` limits selected weight payload before allocation/download. It
does not limit process RSS: the shared table, runtime, allocator, tokenizer,
metadata, scratch, RoPE and session KV are additional. Session/workspace guards
and cancellation ownership remain in force.

## Validation

Tests cover all packed byte values, odd widths and partial blocks, streamed
reader boundaries, finite extremes, independent FP64 dot products, exact
worker/batch invariance, all supported F32/F16/BF16 profile fixtures, GPT-2
transpose/split, tied/separate heads, exact budgets, offline source reuse,
cancellation and independently closed engines sharing one immutable model.
Q4 logits are compared to reconstructed Q4 weights, not required to equal FP32.

The offline real-model probe records FP32/Q8/Q4 logits at identical prefixes,
then records independent eight-token greedy continuations. Within each format,
CLI, HTTP and concurrent SSE must agree. These short probes measure numerical
and transport behavior, not perplexity or general language quality.

```sh
CGO_ENABLED=0 go build -trimpath -o monolith .
go build -o runs/loadtest ./tools/loadtest
python3 -S scripts/q8_reference.py --model PATH_TO_PINNED_MODEL --id gpt2 \
  --binary ./monolith --loadtest runs/loadtest --formats f32 q8 q4 \
  --output runs/q4-gpt2.json
GOMAXPROCS=4 go test -run '^$' -bench '^BenchmarkQ4Linear$' \
  -benchmem -benchtime=250ms -count=3 -cpu=4
```

The script validates source hashes against `validation/hf/model-profiles.json`,
forces offline execution and rotates format order over three trials. Each
fresh server uses four threads/workers, context 128, batch 2, queue 4, KV 32 MiB,
prefill chunk 8, mixed chunk 4 and token budget 16. Two warmup requests precede
eight measured requests at concurrency 2, each generating eight tokens.
RSS is one post-warmup observation per server, not a peak. Startup includes
loading and quantization; the OS file cache is uncontrolled.

The initial GPT-2 probe showed severe drift: both tested next-token argmax IDs
changed; generated text included repeated words. Use FP32/Q8 when this tradeoff
is unsuitable. A throughput improvement does not establish acceptable quality.

## Experiment boundary

We explored activation-dependent bit-plane lookup tables, motivated by the
[T-MAC paper](https://arxiv.org/abs/2407.00088). In local portable-Go prototypes,
building/referencing those tables did not beat Q8 consistently. The shipped
kernel instead uses a small, fixed table to decode packed pairs and retains
floating-point multiplication. It is not T-MAC's algorithm or SIMD backend,
and none of that paper's speedup claims apply here.
