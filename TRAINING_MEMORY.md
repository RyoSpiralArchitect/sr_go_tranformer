# Training memory planning

`train` computes a component memory plan before allocating model tensors, corpus token arrays, backward scratch, or prefetch leases. Use the same settings as the intended run:

```sh
./monolith train -data examples/tiny.txt -preset small \
  -batch 2 -seq 64 -memory-plan
./monolith train -dataset runs/dataset/manifest.json -preset small \
  -batch 2 -seq 64 -data-workers 2 -prefetch 4 -memory-plan
```

The command writes one JSON object to stdout and exits. It creates no checkpoint, metrics file or profile, and starts no training or prefetch goroutines. It still loads the bounded source/metadata and verifies every dataset shard. On resume it scans the checkpoint checksum, reads the saved model/schedule, and checks source identity. It does not tokenize the text corpus or load/check individual tensor values. A successful plan is a capacity check, not validation of text split length, tensor finiteness, output destinations, or training quality.

## Enforcing a budget

```sh
./monolith train -data examples/tiny.txt -preset small \
  -batch 2 -seq 64 -memory-mib 256 -out runs/model.mglm
./monolith train -data examples/tiny.txt -resume runs/model.mglm \
  -memory-mib 256 -out runs/model.mglm
```

`-memory-mib` accepts a nonnegative integer in MiB (1,048,576 bytes). The default `0` disables this additional cap. A run whose planned bytes exceed the cap fails before model/token/workspace allocation or output creation. Equality is accepted; comparison uses integer bytes, without rounding. Negative and overflowing budgets are rejected.

Combining `-memory-plan` with a budget prints the plan even when over budget: `fits_budget` is false and the command exits nonzero. Invalid dimensions, schedules, existing hard allocation limits, IO settings or input integrity produce an error instead of a plan. A larger budget cannot bypass the existing parameter, activation, context, input or prefetch limits.

Ordinary training prints a short estimate to stderr. When `-metrics` is enabled, its start record includes the same plan in `run.memory_plan`. The plan and cap belong to this invocation; they do not change checkpoint bytes or saved training settings. A resumed run can change its budget and dataset IO settings while retaining its saved architecture and schedule.

Resume verification and tensor loading use the same open file descriptor, with budget checking between them. Atomic replacement of the pathname cannot substitute a different checkpoint after inspection. The descriptor closes after tensor loading, before training begins. As before, in-place modification of checkpoint or dataset files is unsupported.

## Accounting

The JSON format is `monolith-training-memory-v1`, with scope `major-array-payload-and-input-staging`. It records model configuration, saved training specification, source shape, individual components, phase estimates and the optional budget.

Let `P` be parameter count, `N = batch × seq`, `D` model dimension, `K` KV dimension, `H` hidden dimension, `V` vocabulary, `L` layers, and `T = seq`. Tensor elements are FP32 unless stated otherwise. The supported target is 64-bit Go.

| Component | Planned bytes |
| --- | ---: |
| Weights | `4P`, including the tied embedding only once |
| Gradients | `4P` |
| Adam moments | `8P` |
| RoPE sine/cosine tables | `4 × context × (D / heads)` |
| Retained backward workspace | `4N × (V + 4D + 3H + 2K)` |
| Training activation bound | `4 × activationElements(config, batch, seq)` |
| Evaluation workspace | `4T × (5D + 2K + 3H + V + heads)` |
| Training tape input copy | `16N`, allowing capacity growth of 64-bit token IDs |

The activation term keeps the existing conservative [forward/backward guard](TRAINING_WORKSPACE.md): `N × [L × (12D + 4K + 6H) + 4V] + 2 × batch × heads × T × L` FP32 equivalents. It includes allocating-backward allowances even though training now retains separate reusable scratch, so it intentionally overestimates that path. The evaluation term assumes a full sequence even when validation has fewer targets.

Input accounting adds these allowances:

| Source | Retained during both phases | Additional during training | Additional during evaluation |
| --- | --- | --- | --- |
| Text with `R` raw bytes | `R + 16 × (R + 4)` for source text and both tokenized splits, including BOS/EOS and capacity growth | `16N` for the sampled X/Y arrays | `8T` for the evaluation input window |
| Dataset with `S` shards, `W` IO workers, depth `Q` | `64NQ + 16384W + 64S` for training prefetch staging, bounded readers and planner arrays | None beyond the tape input copy | `64T + 16384 + 128S` for the separate depth-one validation reader and its two planners |

The dataset staging coefficient is shared with its existing 128 MiB guard and includes token buffers plus span slice growth. Shard count affects planner storage; total corpus token count does not. Manifest/tokenizer objects, channel objects and worker stacks are excluded metadata/overhead. Tokenizer merges can only reduce token count, so the text allowance does not depend on compression ratio.

`persistent_bytes` includes weights, gradients, moments, RoPE, backward scratch and the retained input allowance. Training prefetch remains charged while evaluation runs. The two phases are then:

```text
training_bytes   = persistent_bytes + training_activation_bound_bytes
                   + training_input_allowance_bytes

evaluation_bytes = persistent_bytes + evaluation_workspace_bytes
                   + evaluation_input_allowance_bytes

required_bytes   = max(training_bytes, evaluation_bytes)
```

The estimate counts the maximum of the serial phases. It does not multiply transient arrays by accumulation steps or total optimizer steps. Old arrays may still await collection when the next microbatch begins.

## Scope

This is an estimate of major live array payload and bounded input staging, not a hard heap or process RSS limit. It excludes Go allocator/runtime overhead, object metadata, worker stacks and dispatch allocations, temporary source reading/tokenization, checkpoint/profile IO buffers, and garbage awaiting collection. Source reading and metadata/checksum verification occur before the budget decision. The plan does not set Go's memory limit or alter GC policy; provision additional process headroom and use the existing runtime metrics/profiles to observe real runs.

The CLI budget applies to `train`, including its validation passes and resume. Standalone `eval`, `demo`, inference commands and direct internal model calls retain their existing guards. A successful estimate is not a guarantee against OS allocation failure.

## Validation

The [recorded plan matrix](benchmarks/memory-plan-2026-10-15/plans.json) uses `examples/tiny.txt`, batch 2, sequence 64, byte vocabulary and default training settings:

| Preset | Planned working set (MiB) |
| --- | ---: |
| demo | 2.429 |
| tiny | 8.424 |
| small | 90.082 |
| base | 1293.402 |

These are analytic estimates, obtained without allocating those models. The base plan is rejected with a 1024 MiB budget. The matrix also records batch/sequence scaling, unchanged live memory for accumulation 16, and dataset depth/worker changes.

Tests compare plan components with actual tensor/workspace capacities, cover the byte-exact budget boundary and integer overflow, preserve existing hard guards, and check a case where validation is the larger phase. Text tests cover retained slice capacity with byte and BPE tokenizers. CLI tests inspect/reject a roughly 50 MiB training checkpoint with less than 4 MiB of cumulative allocations, assert no output files are created, reject corruption, and replace a checkpoint path between metadata inspection and tensor loading. See [allocation-check.txt](benchmarks/memory-plan-2026-10-15/allocation-check.txt).

Budget-disabled, budget-enabled, and resumed text/dataset runs produce byte-identical checkpoints, including changed budget and IO settings. The CLI smoke covers the same paths. The full race suite, vet, cgo-free build and smoke pass; [checks](benchmarks/memory-plan-2026-10-15/checks.txt) and [provenance](benchmarks/memory-plan-2026-10-15/provenance.json) record their scope.

The separate [baseline/candidate comparison](benchmarks/memory-plan-2026-10-15/comparison.json) uses merged PR #9 as baseline, two alternating pairs per preset, tiny for 160 steps and small for 40. All four checkpoints per preset match exactly, as do standalone evaluation receipts. Raw timings are retained without a speed claim; this change adds preallocation checks and does not optimize model arithmetic. Synthetic/authored fixtures establish implementation equivalence, not general model quality.

To repeat the comparison, build the baseline and candidate binaries at the recorded revisions/source hash, then run `./scripts/go.sh run ./tools/trainbench` with the arguments in the provenance file. Its prepared dataset can be reused with the exact `train -memory-plan` arguments stored in the plan matrix.
