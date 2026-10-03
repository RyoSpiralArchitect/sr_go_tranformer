# Training measurement

`train` can write bounded, synchronous JSONL records and standard Go pprof files. No monitoring server or dependency is required. Telemetry is separate from `TrainState`; enabling it, changing its output paths, or enabling it only on resume does not change checkpoint bytes for the same successful trajectory.

```sh
mkdir -p runs/measurement
./monolith train -dataset runs/dataset/manifest.json -preset tiny \
  -steps 100 -warmup 5 -seq 64 -batch 2 -accum 1 -threads 4 \
  -eval-every 25 -eval-batches 8 -save-every 50 -log-every 10 \
  -out runs/measurement/model.mglm \
  -metrics runs/measurement/train.jsonl \
  -cpu-profile runs/measurement/cpu.pprof \
  -alloc-profile runs/measurement/alloc.pprof

go tool pprof -top runs/measurement/cpu.pprof
go tool pprof -tags runs/measurement/cpu.pprof
go tool pprof -top -tagfocus=phase=backward runs/measurement/cpu.pprof
go tool pprof -top -sample_index=alloc_space runs/measurement/alloc.pprof
```

Each optional output must be a **new file in an existing directory**. Existing files, duplicate destinations, and aliases of the checkpoint destination through parent symlinks are rejected. Use a fresh set of paths for each invocation, including resume. Startup failure removes only files created by that startup. A metrics write failure at a completed boundary attempts to save that boundary and returns an error. A process crash can leave a partial final JSON line or unfinished profile.

## JSONL contract

Every record has `format: "monolith-train-v1"`, absolute committed `step`/`tokens_seen`, invocation-relative `elapsed_ns`, cumulative `phases`, and memory/IO snapshots. The optional dataset `cursor` is the committed next target, never speculative prefetch progress.

| Event | When / extra fields |
| --- | --- |
| `start` | Before initial validation. `run` records model, saved schedule, source/tokenizer hashes, Go version, thread/IO settings, and either initialization seed or loaded resume-checkpoint hash. |
| `update` | At `-log-every` and the invocation's final planned update. `loss`/`grad_norm` describe that single update, not an interval average. |
| `evaluation` | After successful initial, periodic, and final validation. `loss` and actual `targets` are token weighted. |
| `checkpoint` | After successful periodic/final saves. |
| `end` | After the loop exits. `status` is `completed`, `stopped`, `canceled`, or `failed`. Validation/setup failures before the loop starts may produce no records. |

Records are written by the training owner, without a logger queue or retained history. Their disk volume grows with event count; in-memory telemetry does not. Metrics contain no raw documents or paths. Start and end absolute counters delimit a resumed invocation; subtract its start counters before calculating throughput.

## Timing and memory scope

| Field | Meaning |
| --- | --- |
| `phases.input_ns` | Time spent sampling a text microbatch or waiting for the next ordered dataset lease. |
| `phases.zero_grad_ns` | Clearing or initially allocating gradient arrays. |
| `phases.forward_ns`, `backward_ns` | Training forward and backward wall time across all accumulation microbatches, including any interrupted update. |
| `phases.optimizer_ns` | AdamW validation, clipping, moments and parameter update. |
| `phases.evaluation_ns` | Entire validation call, including its own input reading and forward pass. |
| `phases.checkpoint_ns` | Checkpoint serialization, checksumming, syncing and atomic replacement. |
| `prefetch_read_ns` | Sum of wall time inside training workers' shard range reads, including decoding and failed reads. Concurrent workers overlap each other and compute; **do not add this to elapsed time or the phase sum**. |
| `prefetch_read_tokens` | Successfully read training-prefetch tokens, possibly including unused lookahead. Reused in-buffer ranges do not count as new physical reads. |
| `alloc_bytes`, `gc_cycles` | Process-wide allocation/GC deltas since observer startup, including telemetry, runtime and other goroutines. |
| `heap_bytes` | Current process-wide Go heap allocation; neither tensor footprint nor peak process RSS. |

The phase durations are wall time, not summed CPU time. Their sum excludes miscellaneous orchestration and logging. The prefetch counters are live snapshots and can still advance between records; only the committed cursor measures consumption. `elapsed_ns` starts after source verification/tokenization and model loading, before optimizer-moment setup and the training loop. It includes validation, checkpoint IO and observation overhead. The existing stderr `tokens_per_sec` retains its older scope: cumulative progress since initial validation, including intervening validation and saves.

CPU profiles cover the same invocation plus observer startup/shutdown overhead. Go profiler labels (`phase=input|zero_grad|forward|backward|optimizer|evaluation|checkpoint|prefetch_read`) propagate into kernel goroutines. CPU samples measure CPU activity, so blocked input wait can dominate wall time without dominating CPU samples.

Allocation profiles use Go's default sampling rate and process-wide history, including source/model loading and profiler overhead. A GC is requested after measurement before writing the profile; Go's memory profile can still lag recent allocations/frees. `alloc_space` is cumulative sampled allocation, while `inuse_space` describes sampled retained memory. They are not interchangeable with JSON counters or RSS. Profiles can contain executable and source paths: inspect them before sharing.

Use profiles to select a hotspot. Use separate, unprofiled, repeated runs for performance comparisons, keeping model/data hashes, seed, schedule, batch/sequence/accumulation, threads, IO settings, evaluation/saving intervals and toolchain fixed. Alternate baseline/candidate order and report distributions, allocation counts and numerical checks. A fast kernel microbenchmark alone does not establish faster full training or better held-out quality.

## Validation

Tests compare full checkpoint bytes with telemetry disabled/enabled on both source kinds, including CPU/allocation profiling and resume boundaries. Other checks cover phase/event fields, bounded emission frequency, cancellation, output collisions and cleanup, and saving/resuming after a metrics writer failure. The CLI smoke parses both profiles with Go pprof and compares an observed resumed dataset checkpoint with unobserved continuous training.
