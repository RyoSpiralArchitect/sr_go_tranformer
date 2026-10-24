# Qwen2 tokenizer fixture

The vocabulary is trained only on authored synthetic strings from
`scripts/hf_tokenizer_reference.py`; it contains no pretrained weights.
Generate it with `scripts/hf_qwen_tokenizer_fixture.py --source LOCAL_QWEN_DIR`.
The local source supplies the pinned tokenizer pipeline and chat template from
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775),
copyright the Qwen team, Apache-2.0.

The reference checks NFC, ordered regex splitting, BPE, chat formatting,
special and ordinary added tokens, and lossy UTF-8 decoding. The synthetic
model config includes seven trailing padding IDs outside the tokenizer.
Hashes bind the oracle to the exact tokenizer files. Regenerate with
Tokenizers 0.22.2 and Transformers 4.57.1.
