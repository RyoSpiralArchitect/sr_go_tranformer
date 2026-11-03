#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Replay five real chat turns with prefix reuse off, fixed, and growing."""

import argparse
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import run_measured, sha

# Compare policies within the same production revision: only the fixed binary
# gets the historical borrower guard. Refuse to modify a changed guard.
FIXED = "152bb6fd0c01c93a591cdb61531858000e58740f"
GROWING = "fceaaa737965007751fda507bd4feba36f5be8bf"
SCHEDULE = ("disabled", "fixed", "growing", "growing", "fixed", "disabled")


def capture_method(source):
    start = source.index(b"func (p *prefixSnapshot) capture(")
    end = source.index(b"\n}", start) + 2
    return source[start:end]


def fixed_control(source, old_source, growing_source):
    # The historical policies differ only in the initial borrower guard. Keep
    # the current snapshot storage and replace just that guard in the control.
    def guard(method):
        start = method.index(b"\n") + 1
        end = method.index(b"\n\t}", start) + len(b"\n\t}")
        return method[start:end]

    current = capture_method(source)
    growing = capture_method(growing_source)
    fixed = capture_method(old_source)
    before, after = guard(growing), guard(fixed)
    if (
        before == after or growing.replace(before, after, 1) != fixed
        or guard(current) != before or source.count(current) != 1
    ):
        raise ValueError("capture guard differs from pinned growth policy")
    return source.replace(current, current.replace(before, after, 1), 1)


def check_report(report, mode):
    turns = report["turns"]
    if (
        report["mode"] != mode
        or report["threads"] != 4
        or len(turns) != 5
        or report["build_policy"] != ("fixed" if mode == "fixed" else "growing")
        or report["plan"]["weight_format"] != "q4-mixed"
        or len(report["questions"]) != 5
    ):
        raise ValueError("unexpected mode, geometry or turn count")
    snapshot = []
    retained = 0
    for i, t in enumerate(turns):
        n = len(t["prompt_ids"])
        computed, reused = t["computed_input_tokens"], t["reused_input_tokens"]
        ttft, end = t["first_token_seconds"], t["completion_seconds"]
        prefix, kv = t["retained_prefix_bytes"], t["allocated_kv_bytes"]
        output = t["completion"]["token_ids"]
        if t["stream_ids"] != output or t["stream_text"] != t["completion"]["text"]:
            raise ValueError(f"incomplete displayed reply at {mode} turn {i + 1}")
        if (
            not (0 < computed <= n)
            or computed + reused != n
            or reused < 0
            or reused % 32
            or (i == 0 or mode == "disabled")
            and reused != 0
            or n != t["completion"]["prompt_tokens"]
            or not 1 <= len(output) <= 8
            or len(output) != t["completion"]["completion_tokens"]
            or not (math.isfinite(end) and 0 < ttft <= end)
            or not (0 <= prefix <= kv <= 512 << 20)
            or prefix > (0 if mode == "disabled" else 128 << 20)
        ):
            raise ValueError(f"invalid time, tokens or budget at {mode} turn {i + 1}")
        # Check the *observed* hits against the actual re-encoded transcript.
        # Mistral v0.3 has 8 MiB per 32-token page, 16 pages in the allowance.
        common = 0
        for saved, actual in zip(snapshot, t["prompt_ids"][:-1]):
            if saved != actual:
                break
            common += 1
        if reused != common // 32 * 32:
            raise ValueError(f"prefix hit differs from exact IDs on turn {i + 1}")
        if mode == "growing" or (mode == "fixed" and reused == 0):
            committed = t["prompt_ids"] + output[:-1]
            pages = min(16, len(committed) // 32)
            if pages:
                snapshot = committed[: pages * 32]
                retained = max(retained, pages * (8 << 20))
        if prefix != retained:
            raise ValueError(f"unexpected snapshot payload at {mode} turn {i + 1}")
    return (
        report["plan"],
        report["sampling"],
        report["questions"],
        [(t["prompt_ids"], t["completion"]) for t in turns],
    )


def summary(reports):
    out = {}
    for mode in dict.fromkeys(SCHEDULE):
        trials = [r["report"]["turns"] for r in reports if r["mode"] == mode]
        if len(trials) != 2:
            raise ValueError("missing replicate")
        rows = []
        for turns in zip(*trials):
            row = {}
            for key in (
                "computed_input_tokens",
                "reused_input_tokens",
                "retained_prefix_bytes",
                "allocated_kv_bytes",
            ):
                if turns[0][key] != turns[1][key]:
                    raise ValueError(f"non-deterministic {mode} {key}")
                row[key] = turns[0][key]
            row["first_token_seconds"] = statistics.median(t["first_token_seconds"] for t in turns)
            row["completion_seconds"] = statistics.median(t["completion_seconds"] for t in turns)
            rows.append(row)
        out[mode] = rows
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model, out = args.model.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "status.json").write_text('{"complete":false}\n')
    env = isolated_go_environment()
    pinned = next(
        m
        for m in json.loads((root / "validation/hf/model-profiles.json").read_text())["models"]
        if m["id"] == "mistral-7b-v03"
    )
    inputs = [
        "monolith.go",
        "go.mod",
        "go.sum",
        "prefix_chat_perf_test.go",
        "scripts/compare_prefix_chat.py",
        "scripts/compare_inference.py",
        "scripts/quant_quality.py",
        "validation/hf/model-profiles.json",
    ]
    hashes = {name: sha(root / name) for name in inputs}
    protocol = dict(
        format="monolith-five-turn-chat-v1",
        files=pinned["files"],
        revision=pinned["revision"],
        repo=pinned["repo"],
        schedule=SCHEDULE,
        context=512,
        threads=4,
        max_tokens=8,
        total_kv_mib=512,
        prefix_cache_mib=128,
        hash_inputs=hashes,
        fixed_capture_ref=FIXED,
        growing_capture_ref=GROWING,
        scope="Fresh processes, one sequential conversation per process. TTFT from template "
        "formatting through first token-ID event, excludes model load. Five actual "
        "assistant completions are appended as text. Keep all timings and cache misses.",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def verify_model():
        for name, digest in pinned["files"].items():
            if sha(model / name) != digest:
                raise ValueError("model hash mismatch: " + name)

    def source_at(ref):
        return subprocess.check_output(["git", "show", ref + ":monolith.go"], cwd=root)

    verify_model()
    goroot = subprocess.check_output(
        [root / "scripts/go.sh", "env", "GOROOT"], cwd=root, env=env, text=True
    )
    go = str(Path(goroot.strip()) / "bin/go")
    source = (root / "monolith.go").read_bytes()
    builds = {}
    for name, contents in [
        ("current", source),
        ("fixed", fixed_control(source, source_at(FIXED), source_at(GROWING))),
    ]:
        build = out / name
        build.mkdir()
        for item in inputs[:4]:
            (build / item).write_bytes(
                contents if item == "monolith.go" else (root / item).read_bytes()
            )
        binary = build / "probe.test"
        cmd = [go, "test", "-c", "-trimpath", "-o", binary]
        if name == "fixed":
            module = subprocess.check_output([go, "list", "-m"], cwd=build, env=env, text=True).strip()
            cmd.append("-ldflags=-X=" + module + ".prefixChatBuildPolicy=fixed")
        subprocess.run(cmd + ["."], env=env, cwd=build, check=True)
        expected = "fixed" if name == "fixed" else "growing"
        subprocess.run(
            [binary, "-test.run=^TestChatBuildPolicy$"],
            cwd=build,
            env=dict(env, MONOLITH_EXPECT_CHAT_POLICY=expected),
            check=True,
        )
        builds[name] = {n: sha(build / n) for n in inputs[:4] + ["probe.test"]}
    reports, reference = [], None
    for i, mode in enumerate(SCHEDULE):
        variant = "fixed" if mode == "fixed" else "current"
        output = out / f"{i}-{mode}.json"
        run_env = dict(
            env,
            MONOLITH_CHAT_MODEL=str(model),
            MONOLITH_CHAT_MODE=mode,
            MONOLITH_CHAT_OUTPUT=str(output),
        )
        print("starting", i, mode, flush=True)
        with (out / f"{i}-{mode}.log").open("x") as log:
            resources = run_measured(
                [
                    out / variant / "probe.test",
                    "-test.run=^TestHFMistralGrowingChat$",
                    "-test.v",
                    "-test.timeout=21m",
                ],
                timeout=1280,
                cwd=root,
                outcome_path=out / f"{i}-{mode}-resources.json",
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        report = json.loads(output.read_text())
        journal = [
            json.loads(line) for line in Path(str(output) + ".turns.jsonl").read_text().splitlines()
        ]
        if journal != report["turns"]:
            raise ValueError("turn journal differs from terminal report")
        sig = check_report(report, mode)
        if reference is not None and reference != sig:
            raise ValueError("prompt IDs, transcript, model or completions changed")
        reference = sig
        reports.append(dict(mode=mode, report=report, process_resources=resources))
        print("finished", i, mode, flush=True)
    verify_model()
    if any(sha(root / n) != h for n, h in hashes.items()):
        raise ValueError("source changed during measurement")
    for name, files in builds.items():
        if any(sha(out / name / n) != h for n, h in files.items()):
            raise ValueError("build changed during measurement")
    rows = summary(reports)
    receipt = dict(
        protocol=protocol,
        builds=builds,
        reports=reports,
        summary=rows,
        exact_transcript=True,
        host=platform.system() + "/" + platform.machine(),
        go_version=subprocess.check_output([go, "version"], env=env, text=True).strip(),
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (out / "status.tmp").write_text('{"complete":true}\n')
    (out / "status.tmp").replace(out / "status.json")


if __name__ == "__main__":
    main()
