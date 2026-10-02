# sr_go_tranformer

A self-contained decoder-only Transformer written in Go, with training, text generation, and an HTTP inference server. The implementation uses only the Go standard library and lives in a single source file, [`monolith.go`](monolith.go).

## Features

- RMSNorm, rotary position embeddings (RoPE), grouped-query attention (GQA), SwiGLU, and tied input/output embeddings.
- Manual backpropagation, AdamW, gradient clipping and accumulation, and a warmup/cosine learning-rate schedule.
- Byte-level tokenization with optional byte-pair encoding (BPE).
- Continuous batching through a channel-based inference engine, paged KV caches, bounded queues, and request cancellation.
- Greedy, temperature, top-k, and top-p sampling with repetition penalties.
- Atomic checkpoints containing model weights, optimizer state, tokenizer, and training progress.

## Requirements

- Go 1.22 or later.
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

The server uses a project-specific JSON API. It binds to localhost by default and does not provide authentication, TLS, or streaming responses.

## Development

```sh
go test -race ./...
go vet ./...
./scripts/smoke.sh
go test -run '^$' -bench . -benchmem -cpu 4
```

The tests cover numerical gradients, causal attention, KV-cache equivalence, concurrent inference, cancellation, checkpoint integrity, deterministic resume, and small-model learning. The smoke script exercises the CLI and compares uninterrupted and resumed checkpoints byte for byte.

GitHub Actions runs formatting checks, tests, vet, builds, and the CLI smoke test. See [`VALIDATION.md`](VALIDATION.md) for the recorded experiments and their limits.

## Scope and limitations

This is an experimental foundation for further development. Attention avoids storing a quadratic probability matrix, but dense attention still takes quadratic compute. The KV memory limit covers cache pages, not total process memory. Generation beyond the sequence lengths used during training may degrade quality.

Dataset ingestion and BPE training are currently bounded, in-memory implementations. GPU execution, quantization, distributed training, and compatibility with external pretrained checkpoints are not implemented. Synthetic demo results are implementation checks, not evidence of general language ability.

## License

Copyright 2026 RyoSpiralArchitect.

Licensed under the [Apache License, Version 2.0](LICENSE).
