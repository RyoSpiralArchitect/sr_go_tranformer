#!/usr/bin/env python3
"""Offline Transformers oracle; never imported by the Go runtime.

Install scripts/hf_reference_requirements.txt into an isolated Python 3.11 venv.
Use --fixtures to regenerate CI fixtures, or --model/--binary/--output for a
bounded real-model numerical receipt. Model files must already be local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import safetensors
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

VERSIONS = {"torch": "2.8.0", "transformers": "4.57.1", "safetensors": "0.6.2"}


def versions():
    actual = {k: globals()[k].__version__ for k in VERSIONS}
    if actual != VERSIONS:
        raise RuntimeError(f"Reference versions must be {VERSIONS}; got {actual}")
    return actual


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(architectures=["LlamaForCausalLM"], model_type="llama",
               vocab_size=258, hidden_size=24, intermediate_size=40,
               num_hidden_layers=2, num_attention_heads=6,
               num_key_value_heads=2, max_position_embeddings=128,
               rope_theta=100000.0, rms_norm_eps=1e-5, hidden_act="silu",
               tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
               attention_dropout=0.0, pretraining_tp=1, bos_token_id=1,
               eos_token_id=2, pad_token_id=2, rope_scaling=None)
    torch.manual_seed(132)
    model = LlamaForCausalLM(LlamaConfig(**cfg)).eval()
    model.set_attn_implementation("eager")
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Includes a page crossing, GQA, and two teacher-forced decode positions.
    ids = [1] + [(i * 17 + 41) % 258 for i in range(36)]
    for name, dtype in (("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)):
        folder = root / name
        folder.mkdir(exist_ok=True)
        state = {k: v.to(dtype).contiguous() for k, v in original.items() if k != "lm_head.weight"}
        write_json(folder / "config.json", cfg)
        save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
        loaded = LlamaForCausalLM.from_pretrained(folder, torch_dtype=torch.float32,
                  attn_implementation="eager", local_files_only=True).eval()
        with torch.inference_mode():
            result = loaded(torch.tensor([ids]), use_cache=True)
        cache = []
        for layer in result.past_key_values.layers:
            # HF [batch,kvheads,seq,head_dim] -> Go [seq,kvheads*head_dim].
            cache.append(dict(k=layer.keys[0].transpose(0, 1).flatten(1).tolist(),
                              v=layer.values[0].transpose(0, 1).flatten(1).tolist()))
        write_json(folder / "reference.json",
                   dict(versions=versions(), seed=132, ids=ids,
                        logits=result.logits[0].tolist(), cache=cache,
                        weight_sha256=digest(folder / "model.safetensors")))


def compare_model(folder, binary, output):
    if digest(folder / "model.safetensors") != "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c":
        raise RuntimeError("This receipt targets the pinned SmolLM2-135M-Instruct weights")
    if digest(folder / "config.json") != "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a":
        raise RuntimeError("The pinned model configuration does not match")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = LlamaForCausalLM.from_pretrained(folder, torch_dtype=torch.float32,
               attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
    texts = ["The capital of France is", "Go is a programming language that"]
    cases = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected_ids, expected_logits, past = [], [], None
        inputs = torch.tensor([ids])
        with torch.inference_mode():
            for _ in range(8):
                r = model(inputs, past_key_values=past, use_cache=True)
                logits = r.logits[0, -1].float()
                token = int(logits.argmax())
                expected_ids.append(token)
                expected_logits.append(logits.numpy())
                if token == model.config.eos_token_id:
                    break
                past, inputs = r.past_key_values, torch.tensor([[token]])
        for chunk in (1, 16):
            command = [str(binary.resolve()), "infer-hf", "-hf-model", str(folder),
                       "-ids", ",".join(map(str, ids)), "-context", "512",
                       "-max-tokens", "8", "-prefill-chunk", str(chunk),
                       "-threads", "4", "-logits"]
            got = json.loads(subprocess.check_output(command, text=True))
            a, b = np.asarray(got["logits"], dtype=np.float32), np.asarray(expected_logits)
            error = np.abs(a - b)
            parity = got["token_ids"] == expected_ids
            case = dict(prompt=text, prompt_ids=ids, chunk=chunk,
                        token_ids=got["token_ids"], reference_ids=expected_ids,
                        max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                        greedy_match=parity, text=tokenizer.decode(got["token_ids"]))
            cases.append(case)
            if not parity or not np.isfinite(error).all() or error.max() > 0.002:
                raise RuntimeError(f"Numerical comparison failed: {case}")
    write_json(output, dict(format="monolith-hf-numerical-v1",
               model="HuggingFaceTB/SmolLM2-135M-Instruct",
               revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
               versions=versions(), dtype="source BF16 -> CPU FP32", attention="eager",
               binary_sha256=digest(binary),
               tolerance=0.002, files={p.name: digest(p) for p in sorted(folder.glob("*"))
                                     if p.is_file() and p.name != "README.md"}, cases=cases))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    versions()
    torch.set_num_threads(1)
    if a.fixtures:
        fixtures(a.fixtures)
    elif a.model and a.binary and a.output:
        compare_model(a.model, a.binary, a.output)
    else:
        p.error("provide --fixtures or all of --model --binary --output")


if __name__ == "__main__":
    main()
