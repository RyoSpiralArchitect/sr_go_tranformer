# Dataset pipeline validation

## Scope

The data layer is standard-library Go. These checks establish token equivalence, bounded prefetch ownership and deterministic ordering on local files. They do not measure model quality, training throughput, remote storage latency or production capacity.

Preparation tests cover byte/BPE documents (including UTF-8, invalid bytes, CRLF, blank lines and a final unterminated line), range reads crossing document boundaries, corruption/truncation, invalid vocabulary/framing/metadata, output cleanup and cancellation. Prefetch tests force out-of-order completion and blocked consumers, compare batches and consumed cursors across worker/depth settings, and join workers after cancellation and read failures.

The PR review identified repeated file opens when a microbatch loops through a tiny split many times. Workers now reuse ranges already filled in the same target buffer. A regression test fills 65,536 targets from one- and six-shard tiny corpora, verifies every target, and requires one underlying read per training shard rather than one per repeated visit.

## Memory probe, 2026-10-03

Apple M4, darwin/arm64, Go 1.27.1. One fresh test process per input size; byte tokenizer, 256-byte documents, one training shard plus one validation shard in both cases. Batch 2, sequence 128, workers 2, depth 4; 100 timed microbatches per benchmark. The process also prepares and verifies its dataset, including Go benchmark warmup.

| Raw training input | Incremental live Go heap for prefetch | Peak process RSS | Allocations per microbatch |
| --- | --- | --- | --- |
| 1,048,576 bytes | 17,632 bytes | 20,168,704 bytes | 15 |
| 10,485,760 bytes | 18,944 bytes | 20,791,296 bytes | 15 |

`live_heap_B` is the post-GC heap delta from the opened-dataset baseline to a running pipeline, excluding the already-loaded manifest. RSS is `/usr/bin/time -l`'s maximum resident set size for the entire process. It includes preparation, verification, runtime and IO effects and is not the same as Go heap. This small, single-run probe found no proportional memory growth when payload grew 10× with shard count fixed; metadata still grows with shard count up to the manifest limit. Timing was not used as a speed claim.

Reproduce with a Go test binary, running the two commands separately:

```sh
go test -c -o runs/prefetch.test .
/usr/bin/time -l runs/prefetch.test -test.run '^$' \
  -test.bench '^BenchmarkTokenPrefetchCorpusSize/documents_4096$' -test.benchtime=100x
/usr/bin/time -l runs/prefetch.test -test.run '^$' \
  -test.bench '^BenchmarkTokenPrefetchCorpusSize/documents_40960$' -test.benchtime=100x
```

Create `runs/` first. The timing command above uses macOS options; Linux reports RSS in different units. The structural bound is the fixed number of leased buffers and the prefetch allocation check, not this sample's RSS value.
