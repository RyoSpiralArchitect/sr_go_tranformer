#!/usr/bin/env python3
"""Bounded, offline SmolLM2 CLI/HTTP/SSE comparison against Transformers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def request(url, body=None):
    if body is None:
        with urlopen(url, timeout=60) as r:
            return json.load(r)
    req = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as r:
        if not body.get("stream"):
            return json.load(r)
        ids, text, terminal = [], "", None
        event = ""
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    ids.append(data["token_id"])
                    text += data["text"]
                elif event == "done":
                    terminal = data
                elif event == "error":
                    raise RuntimeError(data)
        if terminal is None or ids != terminal["token_ids"] or text != terminal["text"]:
            raise RuntimeError("SSE events do not match terminal completion")
        return terminal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if torch.__version__ != "2.8.0" or transformers.__version__ != "4.57.1":
        raise RuntimeError("Require Torch 2.8.0 and Transformers 4.57.1")
    if sha(a.model / "model.safetensors") != "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c":
        raise RuntimeError("Require pinned SmolLM2-135M-Instruct weights")
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    model = LlamaForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
    cases = [
        dict(prompt="The capital of France is"),
        dict(messages=[dict(role="user", content="Write one short sentence about cats.")])]
    receipts = []
    common = [str(a.binary.resolve()), "generate", "-hf-model", str(a.model),
              "-context", "512", "-threads", "4", "-max-tokens", "8",
              "-temperature", "0", "-repeat-penalty", "1", "-json"]
    for case in cases:
        if "messages" in case:
            prompt = tok.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
            command = common + ["-chat", "-prompt", case["messages"][0]["content"]]
        else:
            prompt = case["prompt"]
            command = common + ["-prompt", prompt]
        ids = tok.encode(prompt, add_special_tokens=False)
        out_ids, past, inputs = [], None, torch.tensor([ids])
        with torch.inference_mode():
            for _ in range(8):
                result = model(inputs, past_key_values=past, use_cache=True)
                token = int(result.logits[0, -1].argmax())
                out_ids.append(token)
                if token == model.config.eos_token_id:
                    break
                past, inputs = result.past_key_values, torch.tensor([[token]])
        want = dict(text=tok.decode(out_ids, skip_special_tokens=True), token_ids=out_ids,
                    prompt_tokens=len(ids), completion_tokens=len(out_ids),
                    finish_reason="eos" if out_ids[-1] == model.config.eos_token_id else "length")
        actual = json.loads(subprocess.check_output(command, text=True, timeout=120))
        if actual != want:
            raise RuntimeError(f"CLI mismatch: {actual} != {want}")
        receipts.append(dict(request=case, prompt_ids=ids, completion=actual, cli_match=True))
    # The server owns two fixed kernel workers, and all requests use bounded KV.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(a.binary.resolve()), "serve", "-hf-model", str(a.model),
               "-context", "512", "-threads", "4", "-kernel-workers", "2",
               "-max-batch", "2", "-queue", "4", "-kv-mib", "64",
               "-prefill-chunk", "8", "-token-budget", "16", "-addr", f"127.0.0.1:{port}"]
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 40
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("HF server exited during startup")
                try:
                    request(base + "/healthz")
                    break
                except URLError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            for receipt in receipts:
                body = dict(receipt["request"], max_tokens=8, temperature=0, repeat_penalty=1)
                actual = request(base + "/v1/completions", body)
                if actual != receipt["completion"]:
                    raise RuntimeError("HTTP mismatch")
                receipt["http_match"] = True
            def stream(receipt):
                body = dict(receipt["request"], max_tokens=8, temperature=0, repeat_penalty=1, stream=True)
                return request(base + "/v1/completions", body)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for receipt, actual in zip(receipts, pool.map(stream, receipts)):
                    if actual != receipt["completion"]:
                        raise RuntimeError("Concurrent SSE mismatch")
                    receipt["concurrent_sse_match"] = True
            deadline = time.monotonic() + 5
            while True:
                stats = request(base + "/healthz")["stats"]
                if stats["released"] == 4 and stats["admission_used"] == 0 and stats["kv_reserved_bytes"] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Resources not released: {stats}")
                time.sleep(0.02)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError("Server did not stop within shutdown bound")
        if proc.returncode != 0:
            raise RuntimeError(f"Server exit {proc.returncode}")
    a.output.write_text(json.dumps(dict(
        format="monolith-hf-serving-v1", model="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        torch=torch.__version__, transformers=transformers.__version__,
        files={name: sha(a.model/name) for name in ("config.json","model.safetensors","tokenizer.json","tokenizer_config.json")},
        binary_sha256=sha(a.binary), context=512, kernel_workers=2,
        cases=receipts, final_engine_stats=stats, clean_shutdown=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
