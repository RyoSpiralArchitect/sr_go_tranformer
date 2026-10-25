# Synthetic Llama 3.2 Instruct chat oracle

Reuses the authored `hf-llama3-tokenizer` vocabulary and the exact text-only
system/user/assistant path of the pinned Llama 3.2 Instruct template. The model
configuration uses three distinct EOS IDs. No pretrained weights are included.

Generate using `scripts/hf_llama3_chat_fixture.py --source PATH --output
testdata/hf-llama3-chat` in the isolated reference environment. The pinned source
is `mlx-community/Llama-3.2-1B-Instruct-bf16` revision
`863c846a9ac6fad4e49e1743d52984dff262e953`; source metadata hashes are verified.
`--real --output FILE` produces an oracle for the unchanged source tokenizer.

Tokenizers 0.22.2 / Transformers 4.57.1 generate the oracle. The chat date is
explicitly fixed to `25 Oct 2026`; runtime uses the current local date. Cases
include system-only input, multi-turn history, empty text and Python/Jinja trim
semantics. Tools and multimodal/template variations are deliberately unsupported.
