# Synthetic Llama 3 tokenizer oracle

The vocabulary is trained on authored test strings, with all 256 byte symbols
and the pinned Llama 3 special-token names. No pretrained weights or vocabulary
are included. An unreachable whole-word entry exercises `ignore_merges`.
The pipeline and base tokenizer options match
`mlx-community/Llama-3.2-3B-bf16@60a99aaf43164077157d64bf909b7b61143c6a6d`.

Generate with `scripts/hf_llama3_tokenizer_fixture.py --source PATH --output
testdata/hf-llama3-tokenizer` in the isolated reference environment (Tokenizers
0.22.2 / Transformers 4.57.1). `--real --output FILE` instead generates an
oracle for the unchanged pinned source. The source hashes are verified first.

The oracle covers multilingual text, decomposed accents, Unicode numbers,
whitespace, contractions, special tokens, automatic BOS and UTF-8 replacement.
Cleanup outputs are checked across every split point and bytewise streaming.
Long tokenizer-only inputs may exceed the fixture model's context; runtime
generation applies its separate prompt-plus-generation context bound.

`unicode.json` records the pinned Split letter/number/whitespace ranges over all
Unicode scalars. Generate it and the declarations embedded in `monolith.go` with
`scripts/hf_llama3_unicode.py --output testdata/hf-llama3-tokenizer/unicode.json
--go-output runs/unicode-declarations.txt`. Go toolchain upgrades must not change
this tokenizer profile's token IDs.
