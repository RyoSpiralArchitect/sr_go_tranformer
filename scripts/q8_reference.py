#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Bounded offline FP32/Q8/Q4 comparison; Python stdlib only, no downloads.

The numerical reference is this binary's FP32 path, not task quality or Torch.
Compare logits at the SAME prompt, then record independent greedy rollouts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import struct
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def request(url, body=None):
    req = url if body is None else Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        if body is None or not body.get("stream"):
            return json.load(response)
        ids, text, terminal, event = [], "", None, ""
        for raw in response:
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
            raise RuntimeError("SSE events do not match completion")
        return terminal


def compare_logits(a, b, quantized="q8"):
    if len(a) != len(b) or not all(math.isfinite(x) for x in a + b):
        raise RuntimeError("Invalid logits")
    # Stable log-softmax and KL(FP32 || quantized) at one identical input prefix.
    ma, mb = max(a), max(b)
    la = ma + math.log(sum(math.exp(x - ma) for x in a))
    lb = mb + math.log(sum(math.exp(x - mb) for x in b))
    errors = [abs(x - y) for x, y in zip(a, b)]
    argmax = lambda row: max(range(len(row)), key=row.__getitem__)
    return dict(max_abs_error=max(errors), mean_abs_error=sum(errors)/len(errors),
                **{"kl_f32_"+quantized: sum(math.exp(x-la) * ((x-la)-(y-lb)) for x, y in zip(a, b))},
                f32_argmax=argmax(a), **{quantized+"_argmax":argmax(b)}, top1_match=argmax(a) == argmax(b),
                f32_logits_sha256=hashlib.sha256(struct.pack("<%df" % len(a), *a)).hexdigest(),
                **{quantized+"_logits_sha256":hashlib.sha256(struct.pack("<%df" % len(b), *b)).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--id", required=True, help="ID in validation/hf/model-profiles.json")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--loadtest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--formats", nargs="+", choices=("f32", "q8", "q4"), default=["f32", "q8"])
    a = parser.parse_args()
    if len(a.formats) < 2 or a.formats[0] != "f32" or len(set(a.formats)) != len(a.formats):
        parser.error("--formats must start with f32 and contain unique comparison formats")
    formats = tuple(a.formats)
    root = Path(__file__).resolve().parents[1]
    source = next(x for x in json.loads((root / "validation/hf/model-profiles.json").read_text())["models"] if x["id"] == a.id)
    for name, expected in source["files"].items():
        if sha(a.model / name) != expected:
            raise RuntimeError("Pinned source mismatch: " + name)
    env = dict(os.environ, HF_HUB_OFFLINE="1", GOMAXPROCS="4")
    env.pop("HF_TOKEN", None)
    binary = str(a.binary.resolve())

    def run(action, fmt, extra):
        flags = ["-hf-model", str(a.model), "-context", "128"]
        if action != "tokenize-hf":
            flags += ["-weight-format", fmt, "-weight-mib", "8192"]
        return json.loads(subprocess.check_output([binary, action] + flags + extra, env=env, timeout=300))

    plans = {fmt: run("inspect-hf", fmt, []) for fmt in formats}
    requests = source.get("requests", [dict(prompt="The capital of France is"),
                dict(messages=[dict(role="user", content="Write one short sentence about cats.")])])
    cases = []
    for body in requests:
        flags = ["-prompt", body["prompt"]] if "prompt" in body else ["-chat", "-prompt", body["messages"][0]["content"]]
        ids = run("tokenize-hf", "", flags)["token_ids"]
        rows, completions = {}, {}
        for fmt in formats:
            rows[fmt] = run("infer-hf", fmt, ["-ids", ",".join(map(str, ids)), "-max-tokens", "1", "-logits", "-threads", "4"])["logits"][0]
            completions[fmt] = run("generate", fmt, flags + ["-max-tokens", "8", "-temperature", "0", "-repeat-penalty", "1", "-threads", "4", "-json"])
        comparisons = {fmt: compare_logits(rows["f32"], rows[fmt], fmt) for fmt in formats[1:]}
        numerical = comparisons[formats[1]]
        cases.append(dict(request=body, prompt_ids=ids, next_token_comparison=numerical, comparisons=comparisons, completions=completions))
        print(a.id, "prefix", len(cases), "top1", {fmt: c["top1_match"] for fmt, c in comparisons.items()}, flush=True)

    trials = []
    for trial in range(3):
        for fmt in formats[trial % len(formats):] + formats[:trial % len(formats)]:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = "http://127.0.0.1:" + str(port)
            # Equal settings except the selected format/budget. Budget does not
            # change arithmetic; ceil(payload/MiB) shows the smaller load fits.
            budget = math.ceil(plans[fmt]["weight_bytes"] / (1 << 20))
            flags = ["serve", "-hf-model", str(a.model), "-context", "128", "-weight-format", fmt,
                     "-weight-mib", str(budget), "-threads", "4", "-kernel-workers", "4", "-max-batch", "2",
                     "-queue", "4", "-kv-mib", "32", "-prefill-chunk", "8", "-mixed-prefill-chunk", "4",
                     "-token-budget", "16", "-addr", "127.0.0.1:" + str(port)]
            with tempfile.TemporaryFile() as log:
                start = time.monotonic()
                proc = subprocess.Popen([binary] + flags, env=env, stdout=log, stderr=log)
                try:
                    while True:
                        if proc.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(log.read().decode())
                        try:
                            request(base + "/healthz")
                            break
                        except URLError:
                            if time.monotonic() - start >= 300:
                                raise
                            time.sleep(.05)
                    startup = time.monotonic() - start
                    if trial == 0:
                        def complete(case, stream=False):
                            return request(base + "/v1/completions", dict(case["request"], max_tokens=8,
                                temperature=0, repeat_penalty=1, stream=stream))
                        for case in cases:
                            if complete(case) != case["completions"][fmt]:
                                raise RuntimeError("CLI/HTTP mismatch")
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            for case, got in zip(cases, pool.map(lambda c: complete(c, True), cases)):
                                if got != case["completions"][fmt]:
                                    raise RuntimeError("CLI/concurrent SSE mismatch")
                    command = [str(a.loadtest.resolve()), "-url", base, "-concurrency", "2", "-max-tokens", "8",
                               "-short-prompt", "The capital of France is",
                               "-long-prompt", "Once upon a time, a small cat walked through the garden. " * 4,
                               "-timeout", "120s", "-sample-interval", "50ms"]
                    subprocess.run(command + ["-requests", "2"], env=env, check=True, stdout=subprocess.DEVNULL, timeout=300)
                    # ps reports KiB on both supported Unix benchmark hosts.
                    # This is ONE post-warmup observation, never a peak RSS.
                    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(proc.pid)], text=True)) * 1024
                    report = json.loads(subprocess.check_output(command + ["-requests", "8"], env=env, timeout=300))
                    if report["outcomes"] != {"completed": 8} or report["completed_token_events"] != 64:
                        raise RuntimeError("Expected eight complete eight-token requests")
                    stats = request(base + "/healthz")["stats"]
                    if stats["admission_used"] != 0 or stats["kv_reserved_bytes"] != 0:
                        raise RuntimeError("Resources not released")
                finally:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        raise RuntimeError("Shutdown timeout")
                if proc.returncode != 0:
                    raise RuntimeError("Unclean shutdown")
            trials.append(dict(weight_format=fmt, trial=trial+1, weight_mib=budget, startup_seconds=startup,
                post_warmup_rss_bytes=rss, report=report, final_engine_stats=stats, clean_shutdown=True))
            print(a.id, fmt, trial+1, report["completed_token_events_per_second"], "tokens/s", flush=True)
    receipt = dict(format="monolith-low-bit-reference-v1", formats=formats, model=source["repo"], revision=source["revision"],
        files=source["files"], binary_sha256=sha(a.binary), loadtest_sha256=sha(a.loadtest),
        source_sha256=sha(root / "monolith.go"), script_sha256=sha(Path(__file__)),
        host=platform.system()+"/"+platform.machine(), hf_hub_offline=True, plans=plans,
        numerical_reference="same-binary FP32; identical input prefixes; no quality threshold",
        cases=cases, cli_http_concurrent_sse_match=True, warmup_requests=2,
        server=dict(context=128, threads=4, kernel_workers=4, max_batch=2, queue=4, kv_mib=32,
                    prefill_chunk=8, mixed_prefill_chunk=4, token_budget=16),
        rss_method="single post-warmup ps rss KiB observation per server; not peak RSS", trials=trials)
    a.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
