#!/usr/bin/env python3
"""Generate a tiny Qwen2 tokenizer oracle using a local pinned source profile."""
import argparse
import importlib.util
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, trainers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("testdata/hf-qwen-tokenizer"))
    a = p.parse_args()
    assert tokenizers.__version__ == "0.22.2"
    spec = importlib.util.spec_from_file_location("reference", Path(__file__).with_name("hf_tokenizer_reference.py"))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    raw = json.loads((a.source / "tokenizer.json").read_text())
    cfg = json.loads((a.source / "tokenizer_config.json").read_text())
    raw["model"].update(vocab={}, merges=[])
    raw["added_tokens"] = []
    tok = Tokenizer.from_str(json.dumps(raw))
    tok.train_from_iterator(reference.texts(), trainers.BpeTrainer(
        vocab_size=384, min_frequency=1, initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in special])
    tok.add_tokens([AddedToken("<tool_call>", special=False, normalized=False),
                    AddedToken("</tool_call>", special=False, normalized=False)])
    a.output.mkdir(parents=True, exist_ok=True)
    tok.save(str(a.output / "tokenizer.json"))
    actual = json.loads(tok.to_str())
    cfg["added_tokens_decoder"] = {str(t["id"]): {k: v for k, v in t.items() if k != "id"} for t in actual["added_tokens"]}
    cfg["additional_special_tokens"] = special[1:]
    (a.output / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    h = dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2",
             vocab_size=tok.get_vocab_size() + 7, hidden_size=24, intermediate_size=40,
             num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
             max_position_embeddings=512, rope_theta=1000000, rms_norm_eps=1e-6,
             hidden_act="silu", tie_word_embeddings=True, attention_dropout=0.0,
             bos_token_id=tok.token_to_id(special[0]), eos_token_id=tok.token_to_id(special[2]))
    (a.output / "config.json").write_text(json.dumps(h, indent=2) + "\n")
    reference.oracle(a.output, a.output / "reference.json")


if __name__ == "__main__":
    main()
