#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Check every used 275-token KV position and the final vocabulary in two offline builds."""

import argparse
import json
import subprocess
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import run_measured, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "model", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    model, out = args.model.resolve(), args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    env = isolated_go_environment()
    goroot = subprocess.check_output(
        [root / "scripts/go.sh", "env", "GOROOT"], cwd=root, env=env, text=True
    )
    go = str(Path(goroot.strip()) / "bin/go")
    probe = root / "testdata/probes/prompt_kv_test.go"
    files = protocol["model_catalog_entry"]["files"]
    receipt = dict(
        status="incomplete",
        model_files=files,
        probe_sha256=sha(probe),
        protocol_sha256=sha(args.protocol),
        variants={},
    )
    (out / "plan.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def verify_model():
        if any(sha(model / name) != h for name, h in files.items()):
            raise ValueError("model identity mismatch")

    verify_model()
    for variant, ref in [("before", protocol["baseline"]), ("after", protocol["candidate_commit"])]:
        build = out / variant
        build.mkdir()
        hashes = {
            n: protocol["comparison_input_sha256"][n]
            for n in ("go.mod", "go.sum", "inference_perf_test.go")
        }
        hashes.update(
            {"monolith.go": protocol["runtime_sha256"][variant], "prompt_kv_test.go": sha(probe)}
        )
        for name, digest in hashes.items():
            contents = (
                probe.read_bytes()
                if name == "prompt_kv_test.go"
                else subprocess.check_output(["git", "show", ref + ":" + name], cwd=root)
            )
            path = build / name
            path.write_bytes(contents)
            if sha(path) != digest:
                raise ValueError("wrong build input: " + name)
        binary = build / "bits.test"
        subprocess.run(
            [go, "test", "-c", "-trimpath", "-o", binary, "."], cwd=build, env=env, check=True
        )
        binary_hash = sha(binary)
        output = build / "bits.json"
        run_env = dict(env, MONOLITH_BITS_MODEL=str(model), MONOLITH_BITS_OUTPUT=str(output))
        print("starting", variant, flush=True)
        with (build / "run.log").open("x") as log:
            resources = run_measured(
                [binary, "-test.run=^TestMistralPromptBits$", "-test.v", "-test.timeout=13m"],
                env=run_env,
                cwd=build,
                stdout=log,
                stderr=log,
                timeout=800,
            )
        bits = json.loads(output.read_text())
        if (
            bits["positions"] != 275
            or bits["logit_values"] != 32768
            or bits["kv_values"] != 18022400
            or bits["workers"] != 4
            or bits["chunk_rows"] != 16
            or (variant == "before" and bits["token_id"] != 1117)
        ):
            raise ValueError("incomplete or unexpected Mistral forward")
        if sha(binary) != binary_hash or any(sha(build / n) != h for n, h in hashes.items()):
            raise ValueError("build changed during parity check")
        receipt["variants"][variant] = dict(
            report=bits,
            benchmarked=False,
            inputs=hashes,
            binary_sha256=binary_hash,
            resources=resources,
        )
        print("finished", variant, flush=True)
    verify_model()
    passed = receipt["variants"]["before"]["report"] == receipt["variants"]["after"]["report"]
    receipt["status"] = "match" if passed else "mismatch"
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("full prompt numerical mismatch")


if __name__ == "__main__":
    main()
