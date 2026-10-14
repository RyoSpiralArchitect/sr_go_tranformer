# Evaluation without a backward tape

Text and token-dataset evaluation now use a dedicated forward path. It computes the same full-window logits as training, but keeps no `Tape`, inverse RMS values, per-layer attention log-normalizers, or layer activation history. Training's ordinary `Forward` and `Backward` APIs remain available.

## Ownership and arithmetic

Each evaluation invocation owns one workspace and reuses it across all its windows. It stores no input slices or references to training tapes. The model must remain immutable during evaluation; separate evaluation workspaces can safely read the same model concurrently. Parameter updates still require exclusive access.

The workspace has fixed maximum batch and sequence capacities. All token IDs, ownership, configuration, active shape and backing storage are validated before computation. A short final window uses only active prefixes; it never grows or shrinks the retained buffers. Returned logits borrow the active workspace region until its next use, and the evaluation loop consumes them synchronously before releasing the dataset lease. The returned slice's capacity is limited to its active length.

The forward path reuses existing linear, RoPE, RMSNorm, attention and residual-add arithmetic. SwiGLU shares the training expression through `swigluInto`, including its reciprocal-then-multiplication order. Attention still computes the same log-normalizer internally but discards it after use. One score region per batch/head keeps parallel writes disjoint regardless of the number of workers. Every parallel kernel joins before its buffers are reused.

Context cancellation is checked before a window, at layer boundaries, and around the final output projection. Cancellation abandons that window without exposing partial logits. It does not interrupt an individual kernel. The dataset loop releases its lease on both success and error. Workspace contents are scratch and may be overwritten after cancellation; model state is unchanged.

## Retained payload and bounds

There are twelve separate FP32 buffers: five dimension-sized arrays, two KV-sized arrays, three hidden-sized arrays, logits, and per-head attention scores. Their payload is:

```
4 × maxBatch × maxSequence ×
    (5 × dimension + 2 × KV dimension + 3 × hidden + vocabulary + heads) bytes
```

The workspace payload is independent of layer count. Batch 1 / sequence 64 retains 312,832 bytes for tiny and 969,216 bytes for small, plus object and allocator overhead. Text evaluation also owns a reusable input-token array. Dataset input remains governed by its separate prefetch bound.

The constructor checks the payload against the existing 128 Mi-element limit before allocation. Unlike training `Forward`, standalone evaluation no longer charges a backward tape or backward transients, so some shapes rejected by the old evaluation path can now fit. This does not change model parameter or context limits. Evaluation clamps the workspace sequence capacity to the actual number of available targets when the source is shorter than the requested window.

During training validation, backward scratch remains retained. The training guard already charges `backwardWorkspaceElements + activationElements`; its conservative activation estimate exceeds the batch-one evaluation workspace payload for every valid configuration. Evaluation scratch is owned only within each evaluation invocation, so it does not remain live through the next training update. The limit describes managed activation payload, not a process RSS or peak heap ceiling. Runtime objects, parameters, optimizer state, input buffers and unreclaimed garbage remain separate.

## Correctness and measurement

Tests compare every logit and the loss bits with ordinary `Forward` for MHA/GQA/MQA, one/four threads, multiple batches, partial tiles and dirty/NaN buffers. They reuse full → short → full windows, preserve weights/gradients/moments/revision/RoPE and existing training tapes, and run concurrent evaluation with separate workspaces under race detection.

Independent allocating references check complete text and dataset evaluation, including deterministic leading subsets and short final windows. Boundary checks cover invalid inputs/storage, cancellation and reuse, constructor rejection before allocation, and the training/evaluation guard relationship. Existing CLI receipts, full training, telemetry and exact-resume tests remain in force.


## Measurements, 2026-10-14

Apple M4, darwin/arm64, Go 1.27.1, four compute threads. The allocating and workspace paths share model parameters and arithmetic. Three grouped benchmark repetitions of five operations each measure allocation volume; their timing samples are not treated as evidence of a general speedup.

The complete text-evaluation benchmark scores all 1,584 post-BOS targets from the authored validation text with sequence 64. This is the text objective, distinct from the prepared dataset's 1,606 targets with document framing. Each operation includes workspace construction, input framing, every window and loss reduction; model initialization and file reading/tokenization are outside the timer. Median results:

| Preset | Allocating reference, bytes/evaluation | Workspace, bytes/evaluation | Reduction |
| --- | ---: | ---: | ---: |
| tiny | 32,599,830 | 583,344 | 98.2% |
| small | 169,562,460 | 1,369,392 | 99.2% |

A separate warmed forward benchmark uses batch 1 / sequence 64 and excludes construction and loss scoring. Allocating `Forward` uses 1,314,512 bytes/call for tiny and 6,847,216 for small; evaluation uses 10,544 and 15,664 bytes/call. Parallel dispatch still allocates, so this is not an allocation-free path. Construction alone uses 320,896 bytes / 13 allocations for tiny and 977,280 bytes / 13 allocations for small, including allocator/object overhead. Do not add this cost to every warmed window: one evaluation invocation constructs once.

These are cumulative allocation measurements, not live heap or peak RSS. The fixed payload formula above describes the explicitly retained FP32 buffers. Raw benchmark samples and source provenance are retained under [benchmarks/evaluation-2026-10-14](benchmarks/evaluation-2026-10-14).

Baseline source for the separate full-process comparison is merged PR #8, `511ab2f8572b722a389ab0899482d987b2eb66aa`. Source and executable hashes are in [provenance.json](benchmarks/evaluation-2026-10-14/provenance.json). The existing training harness ran five alternating pairs per preset on the prepared dataset, with the same schedule, seeds, threads and IO settings as [the training fixture](TRAINING_VALIDATION.md):

| Preset | Baseline median (range), seconds | Evaluation-workspace median (range), seconds |
| --- | ---: | ---: |
| tiny | 1.489 (1.460–1.506) | 1.520 (1.477–1.943) |
| small | 5.143 (5.064–5.488) | 5.223 (5.039–5.338) |

All twenty timed checkpoints matched by complete SHA-256 within each preset, and dataset evaluation receipts matched byte for byte. Separate text CLI evaluation receipts also matched exactly and are retained beside the report. These checks preserve the successful training/evaluation trajectory; they do not measure general language quality.

No full-training speed improvement is demonstrated: medians are about 2.0% higher for tiny and 1.6% higher for small, and the slower 1.943-second candidate sample is retained. The comparison includes initialization, training, initial/final validation and checkpoint saving, so it has a different scope from the evaluation benchmarks. All [raw timings](benchmarks/evaluation-2026-10-14/comparison.json) remain available. The established improvement is evaluation allocation volume.

## Reproduce

From the candidate checkout, with a matching toolchain and an otherwise quiet machine:

```sh
GOMAXPROCS=4 go test -run '^$' -bench '^BenchmarkEvaluation' \
  -benchtime=5x -count=3 .
mkdir -p runs
evaluation_run=$(mktemp -d runs/evaluation-comparison-XXXXXX)
mkdir "$evaluation_run/baseline-source"
git archive 511ab2f8572b722a389ab0899482d987b2eb66aa | \
  tar -x -C "$evaluation_run/baseline-source"
CGO_ENABLED=0 go -C "$evaluation_run/baseline-source" build -trimpath -o ../baseline .
CGO_ENABLED=0 go build -trimpath -o "$evaluation_run/candidate" .
go run ./tools/trainbench -baseline "$evaluation_run/baseline" \
  -candidate "$evaluation_run/candidate" -out "$evaluation_run/results" -repeats 5
```

To repeat the standalone text receipt check, run both executables with `eval -model` pointing to the same retained checkpoint and `-data examples/heldout/validation.txt -seq 64 -threads 4`, then compare their JSON output byte for byte. The harness already performs the equivalent dataset receipt check.
