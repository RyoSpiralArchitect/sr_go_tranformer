# Synthetic tokenizer fixture

Generated with `scripts/hf_tokenizer_reference.py --fixture testdata/hf-tokenizer`
and Hugging Face Tokenizers 0.22.2. This small BPE is trained on authored test
strings, with the SmolLM2 Digits/ByteLevel pipeline and three special tokens.
One byte-alphabet entry is intentionally absent to exercise null-UNK behavior.
It is not the downloaded SmolLM2 vocabulary.

The reference has deterministic test inputs (seed 135), encoding/decoding IDs,
chat formatting from Transformers, and SHA-256 identities for both tokenizer
files. Go tests require neither Python nor network access.

Repository license: Apache-2.0.
