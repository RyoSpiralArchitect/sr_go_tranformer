# Experimental packed Q4 inference

`-weight-format q4` keeps supported HF matrices in four-bit blocks and uses a
shared 2 KiB table to decode pairs of weights. It is an opt-in memory/throughput
tradeoff. **It changes outputs substantially on some models.** FP32 is still
the default; Q8 is the more conservative quantized option.
For a measured middle ground, [mixed Q4/Q8](QUALITY_INFERENCE.md) protects
embeddings/output heads with Q8 while retaining Q4 projections.

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

## Measured GPT-2 example

On Apple M4 / 24 GiB / Go 1.27.1, with the settings above, three-trial
medians (throughput ranges in parentheses) were:

| Format | Weight payload, bytes | Post-warmup RSS, MiB | Aggregate tokens/s | Startup, s |
| --- | ---: | ---: | ---: | ---: |
| f32 | 497,759,232 | 507.1 | 15.12 (11.14–16.03) | 0.541 |
| q8 | 134,883,888 | 161.8 | 18.63 (18.61–19.68) | 1.110 |
| q4 | 73,117,872 | 101.8 | 21.66 (21.63–22.32) | 1.169 |

These are mixed prefill/decode request measurements, not an isolated decode
kernel. Q4 used about 46% less weight payload than Q8 and its median throughput
was about 16% higher in this workload. Desktop activity, thermal state and
file cache were uncontrolled; timings, especially FP32, varied noticeably.
The comparison includes different quantized numerical results and is not a
measurement of the lookup table alone.

[Raw GPT-2 receipt](validation/q4/gpt2-offline.json) records source revisions,
file hashes, binary/source/script hashes, logits differences, continuations,
request outcomes, resource drain and clean shutdown for every trial.
[Environment](validation/q4/environment.json) identifies the implementation
and host. GPT-2's two next-token KL(FP32 || Q4) values were about 3.91 and 4.17,
versus 0.0147 and 0.0124 for Q8: the memory/speed benefit comes with material
degradation on these inputs.

## Measured Qwen2.5 1.5B example

The same probe and binary also ran the pinned 1,543,714,304-parameter
Qwen2.5-1.5B-Instruct model, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, using its existing local source files.
No new model download or provider inference was used for either probe.

| Format | Weight payload, bytes | Post-warmup RSS, GiB | Aggregate tokens/s | Startup, s |
| --- | ---: | ---: | ---: | ---: |
| f32 | 6,174,857,216 | 5.814 | 1.215 (0.697–1.372) | 3.768 |
| q8 | 1,640,622,080 | 1.553 | 1.616 (1.493–1.717) | 10.386 |
| q4 | 868,837,376 | 0.865 | 1.773 (1.575–2.054) | 8.940 |

Q4 reduced weight payload by about 47% and observed RSS by about 44% relative
to Q8. Median throughput was about 10% higher, but the trial ranges overlap;
this does not establish a consistent speedup. FP32 timings were particularly
variable. Q4 startup was faster than Q8 in this run but still slower than FP32;
packing/quantization has a load-time cost.

Both initial argmax IDs matched FP32, but both independent eight-token
continuations changed. For the cat prompt, FP32/Q8 produced
`Cats are known for their independent nature`, while Q4 produced
`Cats are known for their independence and`. The two next-token KL(FP32 || Q4)
values were about 0.373 and 0.0399; maximum logit errors were 5.71 and 7.15.
This is numerical drift, not evidence of preserved task quality.

[Raw Qwen receipt](validation/q4/qwen25-1.5b-offline.json) includes all nine
server trials. Every format/trial completed eight requests and 64 token events,
released admission and KV reservations, and shut down cleanly. CLI, HTTP and
concurrent SSE matched within each format in the first trial.
The 1.5B run does not validate arbitrary models up to the structural 8B ceiling.

## Projection kernel measurements

The same host and source produced these medians for an 8,960 × 1,536 linear
projection with four fixed workers (milliseconds per call, three samples):

| Input rows | FP32 | Q8 | Q4 |
| ---: | ---: | ---: | ---: |
| 1 | 3.186 | 1.347 | 1.478 |
| 4 | 3.850 | 3.882 | 2.994 |
| 16 | 15.024 | 15.519 | 11.959 |

Q4 was about 10% slower than Q8 for the single-row kernel, while four and
sixteen rows benefited from sharing pair decodes. This is a memory tradeoff
with workload-dependent speed, not a universal decode improvement.
[Raw kernel samples](validation/q4/kernels.txt) also include 3,072 × 768
projections. The Q4 benchmark reports 112 B and one callback allocation per
call; it does not allocate a table per token. Timings do not isolate table
lookup from weight compression, packing and arithmetic changes.

## Experiment boundary

We explored activation-dependent bit-plane lookup tables, motivated by the
[T-MAC paper](https://arxiv.org/abs/2407.00088). In local portable-Go prototypes,
building/referencing those tables did not beat Q8 consistently. The shipped
kernel instead uses a small, fixed table to decode packed pairs and retains
floating-point multiplication. It is not T-MAC's algorithm or SIMD backend,
and none of that paper's speedup claims apply here.
