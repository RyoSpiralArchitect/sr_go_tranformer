# Mixed Q4/Q8 inference and quality probes

`-weight-format q4-mixed` protects token embeddings and the output head with Q8,
while attention and MLP matrices use Q4. It reduces the severe output drift seen
with all-Q4 GPT-2 in our small probe, with a smaller weight payload than all-Q8.
The improvement varies by model and does not recover FP32 behavior. FP32 remains
the default; all-Q8 was closer to FP32 in every distribution comparison below.

```sh
go run . infer local -repo Qwen/Qwen2.5-1.5B-Instruct \
  -revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  -download-mib 4096 -context 128 -weight-format q4-mixed -weight-mib 1000 \
  check -json
HF_HUB_OFFLINE=1 go run . infer local -repo Qwen/Qwen2.5-1.5B-Instruct \
  -revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  -download-mib 4096 -context 128 -weight-format q4-mixed -weight-mib 1000 \
  generate -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 16 -temperature 0 -repeat-penalty 1
```

Source flags go before the action. Direct HF commands, local chat and serving
accept the same format. Native checkpoints and training do not.

## Storage policy

The policy is fixed and explicit, with no calibration step or automatic model
selection. Tied embeddings/output heads occupy one Q8 allocation; an untied
output head has its own Q8 allocation. Norms, biases and GPT-2 position embeddings
remain FP32. Activations, accumulation, RoPE and KV retain the existing FP32
behavior. The [Q8](Q8_INFERENCE.md) and [Q4](Q4_INFERENCE.md) block layouts,
rounding rules and numerical fallback paths are unchanged.

The loader streams directly into the selected per-matrix format, sharing
immutable weights between engines. Mixed models reuse the existing fixed Go
workers, Q4 pair table and bounded workspace. There is no expanded FP32 model,
new kernel, cgo or external inference dependency.

`check -json` reports `q8_parameters`, `q4_parameters`, their total
`quantized_parameters`, scales and the complete selected `weight_bytes`.
`-weight-mib` checks that payload before allocation/download. Runtime, tokenizer,
workspace, allocator overhead and session KV are extra; the budget is not an RSS
limit. Original HF source files and download sizes are unchanged. This is not an
AWQ/GPTQ implementation or importer of prequantized model files.

## Policy selection and held-out evaluation

We first compared six precision assignments on four GPT-2 prefixes. Mean
KL(FP32 || candidate) was 3.698 for Q4, 0.0827 when protecting only the shared
embedding/output matrix, 3.639 for attention-only protection, 3.693 for MLP-only
protection, 0.0660 for embedding plus attention and 0.0186 for all-Q8. Embedding
protection was selected as a small, easily audited policy, then applied unchanged
to the other models. [Selection receipt](validation/quality/gpt2-selection.json).

Evaluation uses six separately authored passages in
[the checked-in corpus](validation/quality/corpus.json). For each document, four
prefix tokens precede sixteen scored next-token positions: 96 positions per
model/format. Each step feeds the **ground-truth previous token**, so every
format sees identical prefixes even when its preferred next token differs.
Passages are plain text, with no chat template. These English passages are a
small held-out probe, not a standardized perplexity benchmark, instruction
following test or general language-quality assessment. Do not compare losses
across different tokenizers/models.

NLL is mean negative log-likelihood of the target token (natural logarithm;
lower is better). KL is mean KL(reference || candidate) over the full next-token
distribution (zero for identical distributions). Top-1 counts identical argmax
IDs. NLL, distribution agreement and independent generated text answer different
questions: a lower NLL alone does not establish better overall generation.

| Model | Format | NLL | KL from FP32 | Top-1 matches / 96 |
| --- | --- | ---: | ---: | ---: |
| GPT-2 | f32 | 3.7773 | 0 | 96 |
| GPT-2 | q8 | 3.7830 | 0.0145 | 81 |
| GPT-2 | q4 | 6.4895 | 3.3304 | 28 |
| GPT-2 | q4-mixed | 3.9287 | 0.1079 | 67 |
| Qwen2.5 1.5B | f32 | 3.4558 | 0 | 96 |
| Qwen2.5 1.5B | q8 | 3.4517 | 0.0017 | 93 |
| Qwen2.5 1.5B | q4 | 3.5125 | 0.3425 | 58 |
| Qwen2.5 1.5B | q4-mixed | 3.4591 | 0.2595 | 67 |
| SmolLM2 360M | f32 | 3.3270 | 0 | 96 |
| SmolLM2 360M | q8 | 3.3391 | 0.0017 | 94 |
| SmolLM2 360M | q4 | 3.7410 | 0.3594 | 53 |
| SmolLM2 360M | q4-mixed | 3.5505 | 0.2270 | 69 |

| Model | FP32 weight bytes | Q8 weight bytes | Q4 weight bytes | Mixed weight bytes |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 | 497,759,232 | 134,883,888 | 73,117,872 | 92,416,560 |
| Qwen2.5 1.5B | 6,174,857,216 | 1,640,622,080 | 868,837,376 | 985,524,224 |
| SmolLM2 360M | 1,447,284,480 | 384,618,240 | 203,738,880 | 227,331,840 |

The GPT-2 mixed payload is about 31.5% smaller than Q8. Its independent
continuation of `The capital of France is` starts
` the capital of the French Republic. The`, while Q4 starts
` a bustling and bustling and bustling. Restaur`. Other continuations still
differ from FP32; removing one repetition example is not proof of preserved
quality. The receipts retain every scored position and both eight-token
continuations, including unfavorable results.

Raw receipts: [GPT-2](validation/quality/gpt2.json),
[Qwen2.5 1.5B](validation/quality/qwen25-1.5b.json),
[SmolLM2 360M](validation/quality/smollm2-360m.json).

## Reproduce offline

Use model files matching the pinned revisions and SHA-256 entries in
[`model-profiles.json`](validation/hf/model-profiles.json). The runner verifies
those files, builds a fresh Go test executable, and starts a separate process
for each format so only one resident model is needed at a time. Python's standard
library orchestrates the experiment; all inference runs in Go.

```sh
python3 -S scripts/quant_quality.py --model PATH_TO_GPT2 --id gpt2 \
  --output-dir runs/quality-gpt2
python3 -S scripts/quant_quality.py --model PATH_TO_QWEN_1_5B --id qwen25-1.5b \
  --output-dir runs/quality-qwen15
python3 -S scripts/quant_quality.py --model PATH_TO_SMOLLM2_360M --id smollm2-360m \
  --output-dir runs/quality-smol360
# Optional: repeat the small selection experiment (1 GiB FP32 weight ceiling).
MONOLITH_ABLATION_MODEL=PATH_TO_GPT2 \
  MONOLITH_ABLATION_OUTPUT=runs/selection.json \
  HF_HUB_OFFLINE=1 GOMAXPROCS=4 go test -run '^TestQualityAblation$' -count=1
```

Choose a fresh output directory and keep source/model files immutable. Reference
vectors remain local; their headers bind model revision, corpus hash, tokenized
inputs, vocabulary, format and row count. The runner rejects different vectors
or changed source files. Receipts include file hashes, binary/build inputs,
source/corpus/script hashes, Go version, per-position loss/KL, memory plan and
raw continuations. The default weight budget is 8 GiB (`--weight-mib` may raise
it to at most 16 GiB), context 128, GOMAXPROCS and kernel
workers 4, at most six documents and 32 steps. The 96-token probes use six
batched sessions; continuation samples run separately.

`score_seconds` includes metric computation and reference I/O, so it is not an
inference throughput benchmark. `post_probe_rss_bytes` is one observation after
scoring/generation, not peak memory. New runner results also include
`process_resources.peak_rss_bytes`: the OS-reported lifetime peak for that one
child, including startup, model loading, scoring and continuations. The runner
uses `wait4` for the individual PID instead of cumulative `RUSAGE_CHILDREN`;
an earlier larger format cannot set the next format's reported peak. Darwin
bytes and Linux KiB are normalized to bytes. This measurement requires macOS or
Linux and is specific to the host/workload, not a memory admission limit.
Historical receipts above retain their original single-observation measurements.
See [Python's wait4 interface](https://docs.python.org/3/library/os.html#os.wait4)
and [Linux resource accounting](https://man7.org/linux/man-pages/man2/getrusage.2.html).

Ordinary CI skips pretrained probes and runs the tiny synthetic
mixed-format loader, budget, numerical, worker, source/offline and engine tests,
plus analytical metric tests. Small subprocess tests verify separate RSS peaks,
nonzero exit handling and timeout kill/reaping on Linux and macOS. No pretrained
weights or reference vectors are checked in.

For the larger model with externally prepared token IDs, see
[the Llama 3.2 3B trial](LLAMA3_INFERENCE.md). That original numerical trial
used external IDs; the linked guide also covers the subsequently verified Go
text tokenizer and Instruct chat path.

## 7B comparison protocol

The pinned source is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Use the original four BF16 shards
whose identities are recorded in the model catalog. The runtime converts them
to the selected Go weight layout through bounded buffers; it does not create
an expanded FP32 model or another quantized weight file on disk.

```sh
HF_HUB_OFFLINE=1 python3 -S scripts/quant_quality.py \
  --model PATH_TO_QWEN_7B --id qwen25-7b \
  --reference q8 --formats q8 q4 q4-mixed --weight-mib 8192 \
  --output-dir runs/qwen7b-quality
HF_HUB_OFFLINE=1 python3 -S scripts/hf_large_model_smoke.py \
  --model PATH_TO_QWEN_7B --id qwen25-7b --format q4-mixed \
  --output-dir runs/qwen7b-serving
```

The quality run uses the same six passages and 96 positions as above, with one
resident format at a time, context 128 and four Go workers. On a 24 GiB host,
the approximately 28.4 GiB FP32 weight payload alone exceeds physical RAM, so
Q8 is the reference. KL and top-1 agreement in this trial are **Q8-relative**;
they cannot establish accuracy relative to the original FP32 model. NLL uses
the same teacher-forced target tokens for every format. Weight payload, peak
RSS and the post-probe observation are reported separately.
The independent FP32 `hf_model_reference.py` runner also preflights its weight
budget before constructing a Torch model. This catalog's 8 GiB 7B budget rejects
that FP32 path early; use the Q8-relative command above on this host.

The optional serving trial compares CLI, HTTP and two concurrent SSE requests
for one resident format, then completes one request and cancels another. It
requires observed server cancellation, drained reservations and clean shutdown.
It is a bounded functional/lifecycle check, not a throughput benchmark. Its
single post-smoke RSS observation is distinct from the quality runner's peak.

## Measured 7B results

On Apple M4 / 24 GiB / Go 1.27.1, the pinned model has 7,615,616,512 parameters
and four original BF16 files totaling 15,231,271,888 bytes. The complete files
were SHA-256 verified before the offline run. The fixed mixed policy and the
original six passages were used unchanged: 96 positions, context 128, four
workers, and one process per format in Q8, Q4, mixed order.

| Format | Weight payload, GiB | Peak RSS, GiB | Mean NLL | KL from Q8 | Top-1 matches / 96 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q8 | 7.537 | 7.099 | 3.5210 | 0 | 96 |
| q4 | 3.991 | 4.200 | 3.8407 | 0.365748 | 65 |
| q4-mixed | 4.498 | 4.172 | 3.7174 | 0.225669 | 72 |

Mixed storage uses 40.3% less weight payload than Q8 and 12.7% more than Q4.
Its mean distribution drift from Q8 is 38.3% lower than Q4 on these positions,
and its top-1 agreement rises from 65 to 72. It still differs from Q8 on 24 of
96 top-1 predictions. FP32 weights would occupy 30,462,466,048 bytes (28.370 GiB)
and were not loaded on this host. These results are Q8-relative and do not
measure error against FP32 or general instruction-following quality.

RSS measures OS residency, separately from allocated weight payload. In this
macOS run the Q8 and mixed peak RSS values are below their allocated payloads,
and mixed's observed peak is slightly below Q4 despite its larger allocation.
These are single-run observations, not evidence that mixed allocates less than
Q4 or a minimum-RAM specification. Use the payload plan plus runtime, tokenizer,
workspace and KV allowances for budgeting. The raw report also retains the
post-probe RSS observation and Go heap allocation; no timings are presented as
a throughput comparison.

For `The capital of France is`, Q8 continues
` Paris. Which of the following statements is`, while both Q4 and mixed continue
` Paris. It is located in the northern`. All three produce
` named Whiskers lived in a cozy` for `Once upon a time, a small cat`.
Both are eight-token, ordinary-text continuations, not chat evaluations.

The [complete 7B quality receipt](validation/quality/qwen25-7b.json) includes
every scored position, all generated IDs, exact weight bytes, per-child peak
RSS, original file hashes and the source/build/probe identities.

The [7B mixed serving receipt](validation/quality/qwen25-7b-serving.json) records
matching CLI/HTTP/concurrent-SSE output for ordinary text and a chat prompt,
then one completed and one canceled load-test request. All six server requests
were released, with zero active/queued/computing requests, admission reservations
and reserved KV bytes at drain. Retained KV pool capacity is recorded separately.
The server shut down cleanly. The chat sample emits
`Cats are independent and fascinating creatures with` and stops at the eight-token
limit; this checks the serving path, not sentence-completion quality.

The serving configuration is context 128, two active sessions, four kernel
workers, a 32 MiB KV pool and a 4,607 MiB weight budget. Its one post-smoke RSS
sample is separate from the quality run's peak. A local model can be checked
and used with a slightly larger weight-payload allowance:

```sh
HF_HUB_OFFLINE=1 ./monolith check-hf -hf-model PATH_TO_QWEN_7B \
  -weight-format q4-mixed -weight-mib 4800 -context 128 -chat -json
HF_HUB_OFFLINE=1 ./monolith generate -hf-model PATH_TO_QWEN_7B \
  -weight-format q4-mixed -weight-mib 4800 -context 128 \
  -chat -prompt 'Write one short sentence about cats.' \
  -max-tokens 8 -temperature 0 -repeat-penalty 1 -json
```
