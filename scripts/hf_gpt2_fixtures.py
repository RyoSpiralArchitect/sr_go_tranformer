#!/usr/bin/env python3
"""Authored GPT-2 numerical and byte-level tokenization oracles; no downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path

import safetensors
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors, trainers


def main():
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, safetensors=safetensors.__version__)
    assert versions == dict(torch="2.8.0", transformers="4.57.1", safetensors="0.6.2")
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    cfg = dict(architectures=["GPT2LMHeadModel"], model_type="gpt2", vocab_size=258,
               n_embd=24, n_inner=40, n_layer=2, n_head=6, n_positions=64, n_ctx=64,
               layer_norm_epsilon=1e-5, activation_function="gelu_new",
               bos_token_id=256, eos_token_id=256)
    for label, dtype in [("legacy-f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        torch.manual_seed(218)
        model = GPT2LMHeadModel(GPT2Config(**cfg)).eval()
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name.endswith(".bias"):
                    value.uniform_(-0.15, 0.15)
        state = {k: v.detach().to(dtype).clone().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
        if label == "legacy-f32":
            state = {k.removeprefix("transformer."): v for k, v in state.items()}
            for i in range(2):
                state[f"h.{i}.attn.bias"] = torch.ones(1, 1, 64, 64).tril()
        folder = root / "testdata/hf-gpt2" / label
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
        loaded = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32,
                     attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
        ids = [256] + [(i*17+41) % 258 for i in range(36)]
        with torch.inference_mode():
            result = loaded(torch.tensor([ids]), use_cache=True)
        cache = [dict(k=layer.keys[0].transpose(0, 1).flatten(1).tolist(),
                      v=layer.values[0].transpose(0, 1).flatten(1).tolist())
                 for layer in result.past_key_values.layers]
        reference = dict(versions=versions, seed=218, ids=ids, logits=result.logits[0].tolist(), cache=cache,
                         weight_sha256=hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest())
        (folder / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False)+"\n")
    spec = importlib.util.spec_from_file_location("reference", root / "scripts/hf_tokenizer_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    folder = root / "testdata/hf-gpt2-tokenizer"
    folder.mkdir(exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=384, min_frequency=1,
          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
          special_tokens=[AddedToken("<|endoftext|>", special=True, normalized=True)], show_progress=False)
    tok.train_from_iterator(ref.texts()+["<|endoftext|>hello<|endoftext|>"], trainer)
    raw = json.loads(tok.to_str())
    # Reproduce the old serializer omissions seen in the pinned public GPT-2.
    raw["model"].pop("type")
    for key in ("pre_tokenizer", "post_processor", "decoder"):
        raw[key].pop("use_regex")
    (folder / "tokenizer.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2)+"\n")
    (folder / "tokenizer_config.json").write_text('{"model_max_length":64}\n')
    config = dict(cfg, vocab_size=tok.get_vocab_size(), bos_token_id=0, eos_token_id=0)
    (folder / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    ref.oracle(folder, folder / "reference.json")


if __name__ == "__main__":
    main()
