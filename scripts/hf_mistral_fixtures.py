#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Seeded full-attention Mistral numerical oracles, with no model download."""
import hashlib
import json
from pathlib import Path

import safetensors
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoModelForCausalLM, MistralConfig, MistralForCausalLM


def main():
    versions = dict(torch=torch.__version__, transformers=transformers.__version__,
                    safetensors=safetensors.__version__)
    assert versions == dict(torch="2.8.0", transformers="4.57.1", safetensors="0.6.2")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1] / 'testdata/hf-mistral'
    cfg = dict(architectures=['MistralForCausalLM'], model_type='mistral',
               vocab_size=258, hidden_size=24, intermediate_size=40,
               num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
               max_position_embeddings=128, rope_theta=1000000.0,
               rms_norm_eps=1e-5, hidden_act='silu', tie_word_embeddings=False,
               attention_dropout=0.0, sliding_window=None,
               bos_token_id=1, eos_token_id=2, pad_token_id=2)
    torch.manual_seed(217)
    model = MistralForCausalLM(MistralConfig(**cfg)).eval()
    for label, dtype in [('f32', torch.float32), ('f16', torch.float16), ('bf16', torch.bfloat16)]:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to(dtype).clone().contiguous() for k, v in model.state_dict().items()}
        (folder / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        save_file(state, folder / 'model.safetensors', metadata={'format': 'pt'})
        loaded = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32,
                    attn_implementation='eager', local_files_only=True, trust_remote_code=False).eval()
        ids = [1] + [(i * 19 + 43) % 258 for i in range(36)]
        with torch.inference_mode():
            result = loaded(torch.tensor([ids]), use_cache=True)
        cache = [dict(k=layer.keys[0].transpose(0, 1).flatten(1).tolist(),
                      v=layer.values[0].transpose(0, 1).flatten(1).tolist())
                 for layer in result.past_key_values.layers]
        reference = dict(versions=versions, seed=217, ids=ids,
                         logits=result.logits[0].tolist(), cache=cache,
                         weight_sha256=hashlib.sha256((folder / 'model.safetensors').read_bytes()).hexdigest())
        (folder / 'reference.json').write_text(json.dumps(reference, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
