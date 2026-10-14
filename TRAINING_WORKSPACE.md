# Trainer-owned backward workspace

The text and token-dataset training loops allocate one fixed-shape workspace per invocation, then reuse it for all layers, microbatches and optimizer updates. The implementation stays in `monolith.go` and uses only the Go standard library. `Model.Backward` retains the allocating path for standalone callers and numerical comparison.

## Ownership and lifetime

The serial training loop owns the workspace. It is bound to one model, its full configuration, and an exact batch/sequence shape. It is not model-global, shared through a pool, or serialized in a checkpoint. Resuming constructs fresh scratch. Training already requires exclusive access to model parameters; the same rule applies to the workspace.

Ten separately allocated slices have explicit roles:

| Buffer | Elements, with N = batch × sequence | Lifetime |
| --- | ---: | --- |
| loss derivative | N × vocabulary | Output projection backward |
| residual gradient | N × dimension | Entire backward pass |
| projection sum | N × dimension | One feed-forward or attention projection group |
| temporary | N × dimension | One projection, RMSNorm result, or attention input gradient |
| query derivative | N × dimension | Attention through query projection |
| hidden, gate and up derivatives | 3 × N × hidden | Feed-forward backward |
| key and value derivatives | 2 × N × KV dimension | Attention through key/value projections |

Linear and attention kernels clear their scratch destinations before accumulation. RMSNorm, SwiGLU and cross entropy overwrite their destinations. Parameter gradients are separate and retain contributions from tied embeddings and accumulation microbatches. Every kernel joins its workers before returning. Neither a forward tape nor a dataset input lease is retained by the workspace, and inputs are released only after synchronous backward completes.

The retained FP32 payload is:

```
4 × batch × sequence × (vocabulary + 4 × dimension + 3 × hidden + 2 × KV dimension) bytes
```

For batch 2 and sequence 64 this is 590,848 bytes for tiny and 1,803,264 bytes for small, independent of layer count. Slice/object and allocator overhead are additional. Parallel worker/closure allocations remain; this is not an allocation-free training loop.

## Budget and compatibility

Before allocating scratch, the constructor adds its payload to the existing conservative activation estimate and checks the same 128 Mi-element guard. The original estimate includes a forward tape and allocating-backward transients, so this combined guard deliberately overestimates the new path and may reject shapes near the previous limit. Training validation uses batch 1 and the same sequence length, which also fits while the workspace is retained.

This is a managed activation guard, not a process memory ceiling. The CLI also checks an optional [startup memory budget](TRAINING_MEMORY.md) that includes parameters, optimizer moments, retained scratch and input staging before tensor allocation. Go runtime overhead and unreclaimed garbage remain outside both estimates.

The architecture, arithmetic order, optimizer, objective, tokenizer, checkpoint format and resume cursor contract are unchanged. Ordinary forward tapes still own their activations. Evaluation now uses a separate [evaluation-only forward workspace](EVALUATION_FORWARD.md), owned within each validation invocation. Its batch-one payload fits inside the existing conservative training activation estimate.

## Correctness checks

Tests compare the allocating and workspace schedules with exact FP32 gradient bits and exact loss bits. They cover all ten buffers prefilled with finite nonzero values or NaNs, nonzero accumulated parameter gradients, repeated token IDs, MHA/GQA/MQA, partial tiles, parallel-sized shapes, and one/four threads. Multiple optimizer updates compare weights, moments and complete checkpoint bytes.

Additional checks reject incompatible owners/configurations/shapes/storage before gradient mutation or tape consumption; preserve two independently retained forward tapes; and reject an over-budget workspace before allocating its payload. A depth-one dataset test cancels during accumulation, verifies the committed checkpoint is unchanged, then reuses the same model and workspace and compares the final checkpoint with an uninterrupted allocating control. Existing text/dataset resume, telemetry and CLI checks remain in force.

## Measurements

Measurements and reproduction instructions follow below. Allocation totals, current heap, retained scratch and wall time have different scopes; none is a peak-RSS measurement.

Apple M4, darwin/arm64, Go 1.27.1, cgo disabled, 2026-10-14. Baseline is main `094c0b83480dd0e158b9d6bfad70e59ae0db34e2`; source/binary/profile hashes are in [provenance.json](benchmarks/workspace-2026-10-14/provenance.json). Both executables use the same [authored holdout](TRAINING_VALIDATION.md), seed 73, batch 2, sequence 64, accumulation 1, four compute threads, two input workers and prefetch depth four.

Separate profiled full runs record process-wide allocation/GC deltas since observer startup, including validation, saving and observation overhead:

| Preset / updates | Baseline allocated bytes | Workspace allocated bytes | Reduction | GC cycles, baseline → workspace |
| --- | ---: | ---: | ---: | ---: |
| tiny / 160 | 923,039,288 | 492,411,016 | 46.7% | 178 → 95 |
| small / 40 | 1,521,131,064 | 948,521,688 | 37.6% | 24 → 15 |

These are single profiled runs per implementation/preset. Raw JSONL counters and sampled allocation summaries are in [the evidence directory](benchmarks/workspace-2026-10-14). Sampled profiles and exact allocation counters have different scopes and need not match. Current heap at the final record was higher in the workspace runs (tiny 5.34 → 9.16 MB; small 76.06 → 103.67 MB); GC timing and retained scratch affect that snapshot. Reduced cumulative allocation does not establish a reduced peak heap or RSS.

The [isolated backward benchmark](benchmarks/workspace-2026-10-14/workspace-bench.txt) uses three repetitions of twenty calls with a fixed tape. Median allocated bytes per call fall from 2,715,405 to 19,872 for tiny and from 14,390,080 to 29,504 for small. Warm allocation counts are 372 and 552 per call, respectively, mostly associated with parallel dispatch. Construction is measured separately: 598,368 bytes / 11 allocations for tiny and 1,810,784 bytes / 11 allocations for small, including allocation overhead. Constructor and warm measurements must not be combined as though construction recurs per microbatch. These grouped microbenchmarks are allocation evidence, not a speed comparison.

Five alternating, unprofiled full-process pairs per preset include source verification, model initialization, initial/final validation and checkpoint saving:

| Preset | Baseline median (range), seconds | Workspace median (range), seconds |
| --- | ---: | ---: |
| tiny | 1.491 (1.483–1.562) | 1.475 (1.458–1.563) |
| small | 4.980 (4.947–5.199) | 5.062 (4.953–6.479) |

There is no demonstrated general speed improvement: tiny's median is about 1.1% lower, while small's is about 1.6% higher, with a slower candidate sample retained in the report. The change targets allocation volume. [All raw timings](benchmarks/workspace-2026-10-14/comparison.json) are retained. All twenty timed checkpoints matched by complete SHA-256 within each preset; standalone baseline/candidate evaluation receipts matched byte for byte. Both profiled checkpoint pairs matched as well. This verifies trajectory preservation on the fixed fixture, not general language quality.

## Reproduce

From the candidate checkout, build baseline and candidate with the same toolchain:

```sh
mkdir -p runs
workspace_run=$(mktemp -d runs/workspace-comparison-XXXXXX)
mkdir "$workspace_run/baseline-source"
git archive 094c0b83480dd0e158b9d6bfad70e59ae0db34e2 | \
  tar -x -C "$workspace_run/baseline-source"
CGO_ENABLED=0 go -C "$workspace_run/baseline-source" build -trimpath -o ../baseline .
CGO_ENABLED=0 go build -trimpath -o "$workspace_run/candidate" .
go run ./tools/trainbench -baseline "$workspace_run/baseline" \
  -candidate "$workspace_run/candidate" -out "$workspace_run/results" -repeats 5
GOMAXPROCS=4 go test -run '^$' -bench '^BenchmarkBackwardWorkspace' \
  -benchtime=20x -count=3 .
```

For profiled allocation runs, use the same training arguments recorded in `comparison.json` and add `-metrics` and `-alloc-profile`, each with a new output path. Keep these runs separate from the unprofiled timing comparison. Raw profile binaries remain local; compact summaries, JSONL counters and their profile hashes are committed.
