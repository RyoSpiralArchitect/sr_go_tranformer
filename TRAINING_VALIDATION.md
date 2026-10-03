# Held-out training and a measured CPU optimization

## Scope and fixed data

The [authored fixture](examples/heldout/README.md) contains 40 distinct training documents and 12 separately written validation documents. No whole line is duplicated within or across the splits. A byte tokenizer is used, so tokenizer fitting cannot see validation text. Preparation uses 2,048-token shards: 4,880 stored training tokens in three shards and 1,607 validation tokens in one shard. Evaluation scores all 1,606 post-leading-BOS validation targets in sequence-64 windows, including the final short window.

This fixture establishes a reproducible development holdout and numerical/performance regression baseline. Its passages are short, authored, related in style, and revisited across epochs. These results do not establish general language ability or performance on an independently sampled natural corpus. Once inspected, this validation split is not a blind final test.

Both models use seed 73, batch 2, sequence 64, accumulation 1, warmup 5, the default AdamW schedule, four compute threads, two IO workers and prefetch depth four. Periodic evaluation/saves are disabled; initial/final validation and the final checkpoint remain enabled. The packed objective, BOS framing, tokenizer, attention and optimizer are unchanged.

| Model | Parameters | Updates / consumed targets | Initial held-out CE | Final held-out CE |
| --- | ---: | ---: | ---: | ---: |
| tiny | 213,696 | 160 / 20,480 | 5.5694320269 | 2.5741678600 |
| small | 4,296,448 | 40 / 5,120 | 5.6597075854 | 2.6718308784 |

Final evaluation receipts and complete checkpoint hashes are in [comparison.json](benchmarks/training-2026-10-03/comparison.json). Baseline and candidate produced byte-identical checkpoints in every measured repetition, and their standalone evaluation receipts matched byte for byte. The falling loss on this narrow holdout is a learning signal for this fixture, not a capability claim.

## Choosing the change

The pre-optimization wall timers put training backward at 1.277 / 1.980 seconds for tiny and 4.270 / 7.179 seconds for small. Small's CPU profile attributed 3.52 and 3.20 sampled seconds to the two `linearBackward` loops, making them the largest application compute sites. See the [phase records](benchmarks/training-2026-10-03/small-profile.jsonl) and [CPU summary](benchmarks/training-2026-10-03/small-cpu-top.txt).

The macOS profiles also contain many native pthread park/signal samples. Their flat percentages should not be interpreted as useful arithmetic utilization or a wall-time breakdown. Selection used the application stacks together with explicit wall timers; the speed comparison below uses unprofiled processes.

`linearBackward` now assigns disjoint groups of four output rows to goroutines. Input gradients share weight loads across four sequence rows; weight gradients share input loads across four output channels. Each element retains the original ascending reduction order, and existing gradients remain accumulated for tied weights and microbatches. Partial tiles use the scalar loop. There is no global scratch state or change to resource ownership.

The numerical oracle tests cover empty/single/partial/full tiles, uneven dimensions, nonzero initial gradients, repeated accumulation, and one/four compute threads. They compare FP32 bits exactly. The existing finite-difference, race, training and exact-resume checks remain in force.

## Matched measurements, 2026-10-03

Apple M4, darwin/arm64, Go 1.27.1, `CGO_ENABLED=0`, `-trimpath`. Baseline source: `99ac08b5eea3ec319ff63fca86b535fc1e6abcee`. Profile source: `a2aec63af3c920d8e58bff3c38260de5f9e0e11e`; its linear kernel is identical to the timing baseline. Exact source and binary hashes are retained in [provenance.json](benchmarks/training-2026-10-03/provenance.json) and the raw comparison.

Five pairs per preset, alternating baseline/candidate order, one fresh process per run. Timing spans the entire training command, including dataset verification, model initialization, initial/final validation and checkpoint saving. No JSONL metrics or profiler is enabled in these timed runs. Standalone evaluation is outside the timed interval. Preparation occurs once, before the repetitions.

| Preset | Baseline median (range), seconds | Candidate median (range), seconds | Median time reduction |
| --- | ---: | ---: | ---: |
| tiny | 2.278 (2.239–2.377) | 1.834 (1.777–2.240) | 19.5% |
| small | 8.861 (8.527–9.220) | 6.964 (6.733–7.622) | 21.4% |

The median of per-pair candidate/baseline time ratios is 0.810 for tiny and 0.791 for small. These are descriptive observations from this laptop and workload, without confidence intervals or a universal speed claim. Thermal/scheduling variation remains visible. Every run's wall, user and system time is retained, including slower samples.

Separate kernel benchmarks used five alternating repetitions of 100 calls, four threads, and identical fixed tensors. Median results (`rows × input × output`):

| Shape | Baseline ns/op | Candidate ns/op | Allocs/op, both |
| --- | ---: | ---: | ---: |
| 128 × 64 × 192 | 347,365 | 202,453 | 13 |
| 128 × 256 × 704 | 4,697,590 | 2,722,643 | 13 |
| 3 × 12 × 258 | 9,212 | 6,317 | 3 |

Raw [baseline](benchmarks/training-2026-10-03/baseline-kernel.txt) and [candidate](benchmarks/training-2026-10-03/candidate-kernel.txt) outputs include allocated bytes. The larger cases remained about 33.8 KB and 131.8 KB per call; the partial-tile case increased from 320 to 336 bytes per call. This PR makes no peak-RSS or allocation-reduction claim. The initial [sampled allocation profile](benchmarks/training-2026-10-03/tiny-alloc-top.txt) remains useful evidence for a later, separate workspace-reuse change.

## Reproduce

Use the same toolchain and an otherwise quiet machine. From the repository root, with the candidate checked out:

```sh
mkdir -p runs
comparison_dir=$(mktemp -d runs/training-comparison-XXXXXX)
mkdir "$comparison_dir/baseline-source"
git archive 99ac08b5eea3ec319ff63fca86b535fc1e6abcee | \
  tar -x -C "$comparison_dir/baseline-source"
CGO_ENABLED=0 go -C "$comparison_dir/baseline-source" build -trimpath -o ../baseline .
CGO_ENABLED=0 go build -trimpath -o "$comparison_dir/candidate" .
go run ./tools/trainbench -baseline "$comparison_dir/baseline" \
  -candidate "$comparison_dir/candidate" -out "$comparison_dir/results" -repeats 5
```

The standard-library Go harness prepares the fixed fixture, alternates execution order, rejects any checkpoint/receipt mismatch, and writes `comparison.json` only after all runs succeed. It retains logs and one checkpoint per implementation/preset; repeated identical checkpoint copies are removed. Use a new result directory for each run. `harness_go_version` identifies the runner; the evaluation receipt also reports the model executable's Go version.

For kernel benchmarks, copy `linear_backward_test.go` into the baseline source, build each root package with `go test -c`, and alternate the two test binaries with:

```sh
GOMAXPROCS=4 path/to/kernel.test -test.run '^$' \
  -test.bench '^BenchmarkLinearBackward$' -test.benchtime=100x
```

Run that pair five times, reversing order on repetitions two and four. For profiles, use the fixed training arguments stored in `comparison.json` and add the [measurement flags](TRAINING_MEASUREMENT.md). CPU/allocation profile binaries remain local because they may contain paths; this repository stores their hashes, compact summaries and JSONL timing records.
