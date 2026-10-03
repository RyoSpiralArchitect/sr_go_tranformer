# Token datasets

`prepare` converts explicitly supplied training and validation text to a versioned binary token dataset. It uses the existing byte/BPE tokenizer and only the Go standard library.

```sh
./monolith prepare -train path/to/train.txt -validation path/to/validation.txt \
  -tokenizer runs/tokenizer.json -out runs/dataset
```

Omit `-tokenizer` for byte tokens. The output directory must not exist. Preparation writes a temporary sibling directory, syncs completed files, and publishes it by rename after success. Failures and cancellation remove temporary output.

## Documents and limits

Each input line is one document, **including its original newline bytes**. A final line without a newline is retained. Blank lines are documents. Arbitrary bytes and CRLF are preserved; there is no normalization. Each document is encoded independently as `[BOS, payload..., EOS]`. This equals calling `Tokenizer.Encode(document, true, true)` on the entire document; it deliberately differs from tokenizing an entire multi-document file in one call. BPE never crosses a document or train/validation boundary.

Preparation reads one bounded document at a time. `-document-bytes` defaults to 65,536 and allows 1..1,048,576 bytes. An oversized document is rejected, never silently split. BPE processing holds intermediate arrays for that document; this is not a bound on total process RSS. Tokenizer training retains its existing 4 MiB limit.

`-shard-tokens` defaults to 1,048,576 and allows 3..16,777,216. Documents cannot span shards or exceed this limit. There are at most 16,384 shards and a 4 MiB manifest. Corpus payload memory does not grow with total tokens; metadata grows with shard count within these limits. Both splits must contain at least one document. The caller is responsible for supplying appropriate separate training/validation data.

## Format v1

`manifest.json` contains `format: "monolith-tokens-v1"`, the full ordered BPE merge table, its SHA-256 identity, vocabulary size, and an ordered shard list. Each entry records a local filename, `train` or `validation`, token/document counts, file size, and SHA-256 of the complete shard. Source paths are not stored. Dataset identity is SHA-256 of the compact Go JSON encoding of the parsed manifest, including shard order and split assignments.

Each `.mgts` file is:

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 8 | ASCII `MGTS0001` |
| 8 | 4 | Vocabulary size, uint32 little endian |
| 12 | 4 | Reserved, all zero |
| 16 | 8 | Token count, uint64 little endian |
| 24 | 8 | Document count, uint64 little endian |
| 32 | 32 | Tokenizer SHA-256 bytes |
| 64 | 4 × tokens | Token IDs, uint32 little endian |

BOS/EOS delimit documents unambiguously: the tokenizer never merges these tokens or emits them as payload. There is no in-memory document index.

## Reader contract

`OpenTokenDataset` verifies all shard sizes, headers, checksums, vocabulary IDs, and complete document framing with a 16 KiB scan buffer before returning. Unknown formats, invalid metadata, duplicate/path-traversing filenames, and non-regular shard files are rejected. Verification is proportional to total bytes on disk.

`ReadTokens` reads a range into caller-owned memory, checks bounds and cancellation, and opens at most one file per call. Reads are limited to 1,048,576 tokens. The dataset holds only bounded manifest/tokenizer metadata; each reader has a buffer of at most 16 KiB. Independent calls can run concurrently. Keep dataset files immutable for the lifetime of readers: per-read size/header checks do not replace the full opening checksum verification or provide snapshot isolation against concurrent file mutation.

The preparation command and reader are the first layer. Prefetch and training/checkpoint integration follow in separate changes; `train -data` continues to use its existing in-memory path.
