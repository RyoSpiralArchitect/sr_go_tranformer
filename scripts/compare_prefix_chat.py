#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Replay five real Mistral chat turns across prefix-cache policies."""

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
TAIL_SCHEDULE = ("growing", "tail", "tail", "growing")
TAIL_LIMITS = dict(computed_ratio_max=0.75, extra_kv_bytes_max=8 << 20, warm_ttft_ratio_max=0.90)


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


class ObservationMismatch(ValueError):
    """A complete trial observed behavior inconsistent with the policy."""


def check_report(report, mode):
    turns = report["turns"]
    if (
        report["mode"] != mode
        or report["threads"] != 4
        or len(turns) != 5
        or report["build_policy"] != ("fixed" if mode == "fixed" else "growing")
        or report.get("tail_enabled", False) != (mode == "tail")
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
        if not (math.isfinite(end) and 0 < ttft <= end):
            raise ValueError(f"invalid timing at {mode} turn {i + 1}")
        if t["stream_ids"] != output or t["stream_text"] != t["completion"]["text"]:
            raise ObservationMismatch(f"incomplete displayed reply at {mode} turn {i + 1}")
        if (
            not (0 < computed <= n)
            or computed + reused != n
            or reused < 0
            or (reused != 0 and (i == 0 or mode == "disabled"))
            or n != t["completion"]["prompt_tokens"]
            or not 1 <= len(output) <= 8
            or len(output) != t["completion"]["completion_tokens"]
            or not (0 <= prefix <= kv <= 512 << 20)
            or prefix > (0 if mode == "disabled" else 128 << 20)
        ):
            raise ObservationMismatch(f"invalid tokens or budget at {mode} turn {i + 1}")
        # Check the *observed* hits against the actual re-encoded transcript.
        # Mistral v0.3 has 8 MiB per 32-token page, 16 pages in the allowance.
        common = 0
        for saved, actual in zip(snapshot, t["prompt_ids"][:-1]):
            if saved != actual:
                break
            common += 1
        expected = common if mode == "tail" else common // 32 * 32
        if reused != expected:
            raise ObservationMismatch(f"prefix hit differs from exact IDs on turn {i + 1}")
        if mode in ("growing", "tail") or (mode == "fixed" and reused == 0):
            committed = t["prompt_ids"] + output[:-1]
            positions = min(512, len(committed))
            if mode != "tail":
                positions -= positions % 32
            if positions:
                snapshot = committed[:positions]
                retained = max(retained, ((positions + 31) // 32) * (8 << 20))
        if prefix != retained:
            raise ObservationMismatch(f"unexpected snapshot payload at {mode} turn {i + 1}")
    return (
        report["plan"],
        report["sampling"],
        report["questions"],
        [(t["prompt_ids"], t["completion"]) for t in turns],
    )


def summary(reports, schedule):
    out = {}
    for mode in dict.fromkeys(schedule):
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
                    raise ObservationMismatch(f"non-deterministic {mode} {key}")
                row[key] = turns[0][key]
            row["first_token_seconds"] = statistics.median(t["first_token_seconds"] for t in turns)
            row["completion_seconds"] = statistics.median(t["completion_seconds"] for t in turns)
            rows.append(row)
        out[mode] = rows
    return out


def tail_result(reports, rows):
    # Aggregate within each real conversation before taking the median of the
    # two conversations. Turn 1 starts with an empty cache in both modes.
    def metrics(mode):
        trials = [r["report"]["turns"] for r in reports if r["mode"] == mode]
        return dict(
            computed=sum(t["computed_input_tokens"] for t in rows[mode]),
            reused=sum(t["reused_input_tokens"] for t in rows[mode]),
            peak_kv_bytes=max(t["allocated_kv_bytes"] for t in rows[mode]),
            warm_ttft_seconds=[sum(t["first_token_seconds"] for t in turn_list[1:]) for turn_list in trials],
        )

    page, tail = metrics("growing"), metrics("tail")
    computed = tail["computed"] / page["computed"]
    extra = tail["peak_kv_bytes"] - page["peak_kv_bytes"]
    warm = statistics.median(tail["warm_ttft_seconds"]) / statistics.median(page["warm_ttft_seconds"])
    return dict(
        page=page, tail=tail, computed_ratio=computed, extra_kv_bytes=extra, warm_ttft_ratio=warm,
        checks=dict(computed=computed <= TAIL_LIMITS["computed_ratio_max"],
                    kv=extra <= TAIL_LIMITS["extra_kv_bytes_max"],
                    warm_ttft=warm <= TAIL_LIMITS["warm_ttft_ratio_max"]),
    )


def finish_comparison(out, receipt):
    reports, schedule = receipt["reports"], receipt["protocol"]["schedule"]
    if [r["mode"] for r in reports] != list(schedule):
        raise ValueError("missing or misordered process reports")
    errors, signatures = [], []
    for i, trial in enumerate(reports):
        try:
            signatures.append(check_report(trial["report"], trial["mode"]))
        except ObservationMismatch as exc:
            errors.append(f"trial {i} ({trial['mode']}): {exc}")

    # Do not aggregate incomparable trajectories. Preserve every completed
    # observation, including a negative verdict, before returning to the CLI.
    receipt["exact_transcript"] = None
    if not errors:
        receipt["exact_transcript"] = all(s == signatures[0] for s in signatures)
        if not receipt["exact_transcript"]:
            errors.append("model, prompt IDs or completions differ between trials")
    if not errors:
        try:
            receipt["summary"] = summary(reports, schedule)
        except ObservationMismatch as exc:
            errors.append(str(exc))
    if not errors and receipt["protocol"]["comparison"] == "tail":
        result = tail_result(reports, receipt["summary"])
        receipt["tail_result"] = result
        errors.extend(f"failed {key} gate" for key, ok in result["checks"].items() if not ok)

    receipt["validation_errors"] = errors
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (out / "status.tmp").write_text(json.dumps(dict(complete=True, accepted=not errors)) + "\n")
    (out / "status.tmp").replace(out / "status.json")
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison", choices=("historical", "tail"), default="historical",
                        help="historical off/fixed/growing or a page-versus-tail ABBA run")
    args = parser.parse_args()
    schedule = TAIL_SCHEDULE if args.comparison == "tail" else SCHEDULE
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
        comparison=args.comparison,
        schedule=schedule,
        context=512,
        threads=4,
        max_tokens=8,
        total_kv_mib=512,
        prefix_cache_mib=128,
        hash_inputs=hashes,
        scope="Fresh processes, one sequential conversation per process. TTFT from template "
        "formatting through first token-ID event, excludes model load. Five actual "
        "assistant completions are appended as text. Keep all timings and cache misses.",
    )
    if args.comparison == "tail":
        protocol["limits"] = TAIL_LIMITS
        protocol["warm_ttft"] = (
            "Sum first-token time over turns 2 through 5 per process, then take the median per mode."
        )
    else:
        protocol.update(fixed_capture_ref=FIXED, growing_capture_ref=GROWING)
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
    variants = [("current", source)]
    if "fixed" in schedule:
        variants.append(("fixed", fixed_control(source, source_at(FIXED), source_at(GROWING))))
    for name, contents in variants:
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
    reports = []
    for i, mode in enumerate(schedule):
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
        reports.append(dict(mode=mode, report=report, process_resources=resources))
        print("finished", i, mode, flush=True)
    verify_model()
    if any(sha(root / n) != h for n, h in hashes.items()):
        raise ValueError("source changed during measurement")
    for name, files in builds.items():
        if any(sha(out / name / n) != h for n, h in files.items()):
            raise ValueError("build changed during measurement")
    receipt = dict(
        protocol=protocol, builds=builds, reports=reports,
        host=platform.system() + "/" + platform.machine(),
        go_version=subprocess.check_output([go, "version"], env=env, text=True).strip(),
    )
    if not finish_comparison(out, receipt):
        raise SystemExit("comparison complete, one or more acceptance checks failed")


if __name__ == "__main__":
    main()
