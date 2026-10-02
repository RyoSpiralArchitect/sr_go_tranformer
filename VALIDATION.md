# Validation — 2026-10-01

Generated checkpoints and raw logs are local artifacts and are not included in this repository. The `runs/` paths below refer to the original validation workspace; use the documented commands to reproduce them.

Environment: Apple M4, macOS/arm64, Go 1.27.1. The binary builds with `CGO_ENABLED=0` and has no Go module dependencies.
Measurements below were taken locally. The final benchmark ran after the training experiment finished; unrelated host activity was not controlled.

## Correctness and concurrency

`go test -race -count=1 -v ./...`: **PASS**, 27 top-level tests plus subtests. `go vet ./...`: **PASS**. `CGO_ENABLED=0 go build -trimpath`: **PASS**.
Raw output: `runs/test-race-final.log`, `runs/vet-final.log`.

- 160 central finite differences across every parameter tensor, including both roles of the tied embedding. Maximum observed absolute derivative discrepancy: **7.31e-5** (FP32, epsilon 2e-4).
- Dense FP64 attention reference, causal masking, isolation of batch rows, and MQA/GQA/MHA configurations.
- Full-forward and cached decoding agree, including page boundaries and sessions at different positions. A separate **4,296,448-parameter**, 6-layer, 8-query/2-KV-head model passes forward/decode parity.
- Independent generation and continuous batching produce identical token sequences for 16 concurrent seeded requests.
- Gradient accumulation matches a combined batch. Single-worker and four-worker forward/backward tensors agree bit for bit.
- Invalid/stale/released/duplicate caches, repeated tape use, and context overflow are rejected.
- Global clipping and an independent first-step AdamW formula, including norm decay exclusion.
- Canceled requests release reservations; closing the engine wakes active and queued callers; admission rejects overload before reading HTTP bodies.
- Checkpoint truncation, extension, checksum corruption, invalid tensor values, and failed atomic saves are covered.

The tests establish these invariants on the exercised configurations. They do not establish production service capacity or training quality at the largest supported size.

## Actual CLI resume

`./scripts/smoke.sh`: **PASS**.

This trains a BPE tokenizer from `examples/tiny.txt`, trains for 12 AdamW updates with two accumulated microbatches, and compares that result with a process stopped after 5 updates and resumed for the remaining 7.
The **entire checkpoint files**, including optimizer moments, schedule, RNG state and checksum, are byte-identical:

```text
4745596f438530bb82bcd65ae812297eb0a4f2a7874ada64fbe80bf0caa998a4
```

The same script successfully exercises `eval`, `generate -json`, and `inspect` through the compiled binary.
It leaves its receipts in `runs/cli-smoke/`. The test is for execution and reproducibility; its 12-update model is not a language-quality result.

## Synthetic learning experiment

```sh
./scripts/go.sh run . demo -out runs/demo-framed-500.mglm
```

- 32,992 parameters; random initialization; byte vocabulary 258.
- 500 updates × 2 sequences × 96 targets = **96,000 supervised targets**.
- Peak learning rate 0.005; final 0.001; 5-update warmup; seed 7 for initialization, seed 77 for sampling.
- A 10% trailing raw-byte split, separated before tokenization.
- Deterministic validation subset: first 8 windows, **768 targets**.
- Validation cross-entropy: **5.533675 → 0.090957** nats/token.
- Final sampled training loss: **0.080779**.

Greedy continuation, repetition penalty 1:

```text
the spiral remembers the rain.
the cat follows the spiral.
the rain returns to the sea.
th
```

Both the train and validation portions contain repetitions of the same eight authored sentences. This is an overfitting/implementation check, **not evidence of general language ability or held-out topic generalization**. The final `th` is the explicit token-length stop.

Checkpoint SHA-256:

```text
771c23071136ff2905241941a777cc055f4ba82e8a6d77da5bf55288171c224d
```

Raw receipts: `runs/demo-framed-500-training.log`, `runs/demo-framed-500-output.txt`.
Running `./monolith demo` creates `runs/demo.mglm` for the README commands.
A separate unit test learns the repeated `abc` next-token pattern: **5.562956 → 0.146582**, including the uncertain first prediction from BOS.

## Attention memory growth

Command:

```sh
./scripts/go.sh test -run '^$' \
  -bench 'BenchmarkDecodeBatch|BenchmarkAttentionLinearMemory' \
  -benchmem -benchtime=300ms -count=1 -cpu=4
```

One attention layer, dimension 32, 4 query heads, 2 KV heads. Inputs are allocated outside the timed loop; outputs, normalizers and scratch are included.

| Sequence length | Allocated bytes/op | Allocs/op | ns/op |
|---:|---:|---:|---:|
| 128 | 22,993 | 12 | 175,292 |
| 256 | 45,504 | 12 | 640,697 |
| 512 | 90,561 | 12 | 2,481,265 |
| 1024 | 180,672 | 12 | 9,615,449 |

An 8× longer sequence uses approximately **7.86×** as many allocated bytes in this kernel. Dense causal attention still performs quadratic work. These bytes/op are allocation measurements, **not total process RSS or a complete training peak-memory measurement**.

## Batched decode kernel

Same local benchmark run. The 32,992-parameter demo model, 32 cached tokens per independent session, one new token per session, reusable workspaces/pages. Sampling and HTTP are excluded.

| Sessions per batch | Tokens/op | ns/op | Allocated bytes/op | Allocs/op |
|---:|---:|---:|---:|---:|
| 1 | 1 | 18,590 | 2,928 | 20 |
| 4 | 4 | 45,158 | 6,648 | 20 |
| 8 | 8 | 89,150 | 11,279 | 20 |

This is one local microbenchmark, with no external baseline. It is not a throughput claim for the 75M preset, an HTTP load test, or a general Go-versus-other-language comparison.
Raw output: `runs/benchmarks-final.txt`.

## Scope remaining

The 75,714,816-parameter preset is configuration-validated, but was not trained or benchmarked here. No billion-parameter, GPU, distributed, production-load, or pretrained-model compatibility claim is made. Dataset ingestion and BPE training remain bounded in-memory implementations.
