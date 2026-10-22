#!/usr/bin/env python3
"""Small seeded, offline numerical oracles for additional inference profiles."""
import hashlib
import json
from pathlib import Path

import safetensors
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM


def main():
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, safetensors=safetensors.__version__)
    assert versions == dict(torch="2.8.0", transformers="4.57.1", safetensors="0.6.2")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1] / "testdata"
    profiles = [("qwen2", "f32", True, torch.float32),
                ("qwen2", "f16", True, torch.float16),
                ("qwen2", "untied-bf16", False, torch.bfloat16),
                ("llama", "untied-f32", False, torch.float32)]
    for family, label, tied, dtype in profiles:
        cfg = dict(architectures=["Qwen2ForCausalLM" if family == "qwen2" else "LlamaForCausalLM"],
                   model_type=family, vocab_size=258, hidden_size=24, intermediate_size=40,
                   num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
                   max_position_embeddings=128, rope_theta=100000.0, rms_norm_eps=1e-5,
                   hidden_act="silu", tie_word_embeddings=tied, attention_dropout=0.0,
                   bos_token_id=1, eos_token_id=2, pad_token_id=2, rope_scaling=None)
        if family == "qwen2":
            cfg.update(use_sliding_window=False, sliding_window=None, max_window_layers=2)
        torch.manual_seed(216)
        model = (Qwen2ForCausalLM(Qwen2Config(**cfg)) if family == "qwen2"
                 else LlamaForCausalLM(LlamaConfig(**cfg))).eval()
        # Default Qwen initialization zeros biases, which would hide a missing add.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith(".bias"):
                    p.uniform_(-0.15, 0.15)
        folder = root / ("hf-" + family) / label
        folder.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to(dtype).clone().contiguous() for k, v in model.state_dict().items()
                 if not (tied and k == "lm_head.weight")}
        (folder / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
        loaded = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32,
                    attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
        ids = [1] + [(i * 17 + 41) % 258 for i in range(36)]
        with torch.inference_mode():
            result = loaded(torch.tensor([ids]), use_cache=True)
        cache = [dict(k=layer.keys[0].transpose(0, 1).flatten(1).tolist(),
                      v=layer.values[0].transpose(0, 1).flatten(1).tolist())
                 for layer in result.past_key_values.layers]
        reference = dict(versions=versions, seed=216, ids=ids, logits=result.logits[0].tolist(),
                         cache=cache, weight_sha256=hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest())
        (folder / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
