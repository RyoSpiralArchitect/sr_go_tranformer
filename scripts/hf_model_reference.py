#!/usr/bin/env python3
"""Offline, hash-bound numerical and serving checks for the model catalog."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import gc
import json
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from urllib.error import URLError

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("serving", Path(__file__).with_name("hf_serving_reference.py"))
    serving = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serving)
    assert torch.__version__ == "2.8.0" and transformers.__version__ == "4.57.1"
    source = next(x for x in json.loads((root / "validation/hf/model-profiles.json").read_text())["models"] if x["id"] == a.id)
    for name, expected in source["files"].items():
        if serving.sha(a.model / name) != expected:
            raise RuntimeError("Pinned model identity mismatch: " + name)
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
    binary = str(a.binary.resolve())
    budget = source.get("weight_mib", 3072)
    flags = ["-hf-model", str(a.model), "-context", "512", "-weight-mib", str(budget), "-threads", "4"]
    requests = source.get("requests", [dict(prompt="The capital of France is"),
                dict(messages=[dict(role="user", content="Write one short sentence about cats.")])])
    expected = []
    eos = model.config.eos_token_id
    stops = eos if isinstance(eos, list) else [eos]
    for body in requests:
        text = body.get("prompt")
        if text is None:
            text = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens='prompt' in body)
        expected_ids, logits, past = [], [], None
        inputs = torch.tensor([ids])
        with torch.inference_mode():
            for _ in range(8):
                r = model(inputs, past_key_values=past, use_cache=True)
                row = r.logits[0, -1].float()
                token = int(row.argmax())
                logits.append(row.numpy())
                expected_ids.append(token)
                if token in stops:
                    break
                past, inputs = r.past_key_values, torch.tensor([[token]])
        want = dict(text=tok.decode(expected_ids, skip_special_tokens=True), token_ids=expected_ids,
                    prompt_tokens=len(ids), completion_tokens=len(expected_ids),
                    finish_reason="eos" if expected_ids[-1] in stops else "length")
        expected.append((body, text, ids, expected_ids, logits, want))
    # Finish the independent reference before loading any Go model weights.
    del model, r, past, inputs
    gc.collect()
    cases = []
    for body, text, ids, expected_ids, logits, want in expected:
        numerical = []
        for chunk in (1, 16):
            cmd = [binary, "infer-hf"] + flags + ["-ids", ",".join(map(str, ids)), "-max-tokens", "8",
                  "-prefill-chunk", str(chunk), "-logits"]
            got = json.loads(subprocess.check_output(cmd, text=True, timeout=240))
            error = np.abs(np.asarray(got["logits"], dtype=np.float32) - np.asarray(logits))
            if got["token_ids"] != expected_ids or not np.isfinite(error).all() or error.max() > 0.002:
                raise RuntimeError(f"Numerical mismatch chunk={chunk} max_abs_error={error.max()}")
            numerical.append(dict(chunk=chunk, max_abs_error=float(error.max()), greedy_match=True))
        cmd = [binary, "generate"] + flags + ["-max-tokens", "8", "-temperature", "0", "-repeat-penalty", "1", "-json"]
        cmd += ["-prompt", body["prompt"]] if "prompt" in body else ["-messages", "-"]
        data = None if 'prompt' in body else json.dumps(body['messages'])
        actual = json.loads(subprocess.check_output(cmd, input=data, text=True, timeout=240))
        if actual != want:
            raise RuntimeError(f"CLI mismatch: {actual} != {want}")
        cases.append(dict(request=body, formatted_prompt=text, prompt_ids=ids, completion=actual, numerical=numerical, cli_match=True))
        print(a.id, "numerical and CLI case passed", len(cases), flush=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    cmd = [binary, "serve"] + flags + ["-kernel-workers", "2", "-max-batch", "2", "-queue", "4",
           "-kv-mib", "128", "-prefill-chunk", "8", "-token-budget", "16", "-addr", f"127.0.0.1:{port}"]
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Server failed to start")
                try:
                    serving.request(base + "/healthz")
                    break
                except URLError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            for case in cases:
                body = dict(case["request"], max_tokens=8, temperature=0, repeat_penalty=1)
                if serving.request(base + "/v1/completions", body) != case["completion"]:
                    raise RuntimeError("HTTP mismatch")
                case["http_match"] = True
            def stream(case):
                return serving.request(base + "/v1/completions",
                     dict(case["request"], max_tokens=8, temperature=0, repeat_penalty=1, stream=True))
            with ThreadPoolExecutor(max_workers=2) as pool:
                for case, actual in zip(cases, pool.map(stream, cases)):
                    if actual != case["completion"]:
                        raise RuntimeError("Concurrent SSE mismatch")
                    case["concurrent_sse_match"] = True
            deadline = time.monotonic() + 5
            while True:
                stats = serving.request(base + "/healthz")["stats"]
                if stats["released"] == len(cases)*2 and stats["admission_used"] == 0 and stats["kv_reserved_bytes"] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Serving resources not released")
                time.sleep(0.02)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError("Shutdown timed out")
        if proc.returncode != 0:
            raise RuntimeError("Server did not shut down cleanly")
    receipt = dict(format="monolith-hf-model-reference-v1", model=source["repo"], revision=source["revision"],
                   files=source["files"], binary_sha256=serving.sha(a.binary), context=512, weight_mib=budget,
                   torch=torch.__version__, transformers=transformers.__version__, tolerance=0.002,
                   attention="eager", dtype=source.get("dtype", "source BF16 -> CPU FP32"), cases=cases,
                   final_engine_stats=stats, clean_shutdown=True)
    a.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(a.id, "HTTP, concurrent SSE and resource release passed", flush=True)


if __name__ == "__main__":
    main()
