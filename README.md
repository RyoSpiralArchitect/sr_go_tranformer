# sr_go_tranformer

A self-contained decoder-only Transformer written in Go, with training, text generation, and an HTTP inference server. The implementation uses only the Go standard library and lives in a single source file, [`monolith.go`](monolith.go).

## Features

- RMSNorm, rotary position embeddings (RoPE), grouped-query attention (GQA), SwiGLU, and tied input/output embeddings.
- Manual backpropagation, AdamW, gradient clipping and accumulation, and a warmup/cosine learning-rate schedule.
- Byte-level tokenization with optional byte-pair encoding (BPE).
- Continuous batching with bounded token budgets, chunked prefill, paged KV caches, and request cancellation.
- SSE token streaming with bounded per-request buffers and slow-consumer isolation.
- Greedy, temperature, top-k, and top-p sampling with repetition penalties.
- Atomic checkpoints containing model weights, optimizer state, tokenizer, and training progress.

## Requirements

- Go 1.22 or later. On macOS 26, use a current Go release.
- A 64-bit system.

The project runs on the CPU using FP32 tensors. It does not require cgo, Python, or an external machine-learning library.

## Quick start

```sh
git clone https://github.com/RyoSpiralArchitect/sr_go_tranformer.git
cd sr_go_tranformer

go test ./...
go build -o monolith .
./monolith demo
```

The demo trains a small model for 500 updates on a built-in synthetic corpus, generates a continuation, and saves `runs/demo.mglm`. No dataset or model download is needed.

```sh
./monolith generate -model runs/demo.mglm \
  -prompt 'the spiral' -max-tokens 60 -temperature 0 -repeat-penalty 1
```

## Training

Train a model from a text file:

```sh
./monolith train -data examples/tiny.txt -preset demo \
  -steps 500 -seq 96 -batch 2 -out runs/model.mglm
```

Available presets are `demo`, `tiny`, `small`, and `base`. JSON files in [`configs/`](configs/) show how to customize model dimensions using `-config`. Model weights are initialized randomly; pretrained weights are not included.

Training reserves the trailing 10% of the input bytes for validation before tokenization. Use separate, unused text for independent evaluation:

```sh
./monolith eval -model runs/model.mglm -data path/to/held-out.txt -seq 96
```

To pause after a fixed number of updates and resume the original schedule:

```sh
./monolith train -data examples/tiny.txt -preset demo \
  -steps 500 -seq 96 -stop-after 100 -out runs/resume.mglm
./monolith train -data examples/tiny.txt \
  -resume runs/resume.mglm -out runs/resume.mglm
```

Checkpoints preserve optimizer and random-number-generator state. Resume requires the same corpus and retains the original model, tokenizer, and learning-rate schedule.

For BPE, train a tokenizer on training-only text and pass its JSON file with `-tokenizer`:

```sh
./monolith tokenizer -data path/to/training-only.txt \
  -vocab 512 -out runs/tokenizer.json
```

Use `./monolith <command> -h` to see all options. Commands include `demo`, `tokenizer`, `train`, `eval`, `generate`, `inspect`, and `serve`.

## HTTP inference

```sh
./monolith serve -model runs/demo.mglm \
  -addr 127.0.0.1:8080 -threads 4 -max-batch 8 -queue 32 -kv-mib 64
```

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"the spiral","max_tokens":60,"temperature":0,"repeat_penalty":1}'
```

Responses contain `text`, `token_ids`, `prompt_tokens`, `completion_tokens`, and `finish_reason`. Health and Prometheus-format metrics are available at `/healthz` and `/metrics`.

The server uses a project-specific JSON API and binds to localhost by default. Authentication and TLS are not built in.

To receive tokens as they are generated, add `"stream": true`:

```sh
curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"the spiral","max_tokens":60,"temperature":0,"repeat_penalty":1,"stream":true}'
```

The response uses Server-Sent Events (SSE):

- `token`: `{ "token_id": 116, "text": "t" }`, once per generated token. Text may be empty while a UTF-8 character is incomplete, or for a special token.
- `done`: the complete response with text, token IDs, usage counts, and finish reason. Concatenating all token-event text gives the final text, including replacement characters for invalid bytes.
- `error`: `{ "error": "...", "code": "..." }` if an accepted stream fails. Check this event even when the HTTP status is 200. Invalid input or immediate overload returns a normal JSON error before streaming begins.

One goroutine owns sessions and KV pages. HTTP handlers drain bounded event channels; if a channel fills, the engine terminates that request with `slow_consumer` and releases its KV reservation. The terminal result uses a separate mailbox, so a slow reader cannot block the engine. A disconnected client cancels its request. Streams have a two-minute request timeout, a five-second limit per network write/flush, and ten-second keep-alive comments during idle periods.

### Scheduling and capacity

```sh
./monolith serve -model runs/demo.mglm -threads 4 \
  -max-batch 8 -queue 32 -kv-mib 64 \
  -prefill-chunk 16 -mixed-prefill-chunk 1 -token-budget 64 -stream-buffer 32
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-max-batch` | 8 | Maximum active sessions. |
| `-queue` | 32 | Pending request capacity. |
| `-kv-mib` | 256 | KV page budget; full request capacity is reserved before execution. |
| `-prefill-chunk` | 16 | Maximum prompt tokens per session per scheduling tick. Set 1 for sequential prefill. |
| `-mixed-prefill-chunk` | 1 | Prompt tokens per session when any session is already generating; capped by `prefill-chunk`. |
| `-token-budget` | `max(max-batch, 64)` | Total tokens processed in a tick, between `max-batch` and 1,024. |
| `-stream-buffer` | 32 | Pending token events per stream, between 1 and 4,096. |

Each active session receives at least one token of work every tick. Remaining capacity is shared among prompt chunks. By default, prompt chunks shrink to one token while any session is generating, preserving a shorter scheduling tick for ongoing output. Set `-mixed-prefill-chunk` equal to `-prefill-chunk` to favor prompt throughput over output cadence. Chunked prefill batches matrix projections across prompt rows and computes vocabulary logits only for each chunk's final row. The token budget bounds rows of work, not wall-clock latency; attention cost still grows with context length.

### Measurement

The metrics endpoint includes queue-wait, engine first-token, and engine token-gap histograms, processed token counts, KV use, heap allocation, GC counters, and goroutine count. Engine timing ends when a token is sampled; client timing also includes HTTP and delivery. Heap metrics are not process RSS.

A separate standard-library load client checks stream output and measures mixed requests:

```sh
go run ./tools/loadtest -requests 60 -concurrency 4 -max-tokens 16 \
  -cancel-every 7 -slow-every 5 -details > runs/loadtest.json
```

Create `runs/` first if needed. See the [load client documentation](tools/loadtest/README.md) for timing definitions and [serving validation](SERVING_VALIDATION.md) for a measured comparison of sequential and chunked prefill.

## Development

```sh
go test -race ./...
go vet ./...
./scripts/smoke.sh
go test -run '^$' -bench . -benchmem -cpu 4
```

The tests cover numerical gradients, causal attention, mixed-length chunked prefill, KV-cache equivalence, streamed output and UTF-8 boundaries, slow consumers, cancellation and shutdown, checkpoint integrity, deterministic resume, and small-model learning. The smoke script exercises the CLI and compares uninterrupted and resumed checkpoints byte for byte.

GitHub Actions runs formatting checks, tests, vet, builds, and the CLI smoke test with Go 1.22 on Linux and the current stable Go release on Linux and macOS. See [`VALIDATION.md`](VALIDATION.md) for the recorded experiments and their limits.

## Scope and limitations

This is an experimental foundation for further development. Attention avoids storing a quadratic probability matrix, but dense attention still takes quadratic compute. The KV memory limit covers cache pages, not total process memory. Generation beyond the sequence lengths used during training may degrade quality.

Dataset ingestion and BPE training are currently bounded, in-memory implementations. GPU execution, quantization, distributed training, and compatibility with external pretrained checkpoints are not implemented. Synthetic demo results are implementation checks, not evidence of general language ability.

## License

Copyright 2026 RyoSpiralArchitect.

Licensed under the [Apache License, Version 2.0](LICENSE).
