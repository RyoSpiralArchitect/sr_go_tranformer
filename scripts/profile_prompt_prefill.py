#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Capture a post-load Mistral prompt in separate CPU, trace and plain processes."""
import argparse
import json
import platform
import subprocess
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import run_measured, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model, out = args.model.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = isolated_go_environment()
    source = next(m for m in json.loads((root / 'validation/hf/model-profiles.json').read_text())['models']
                  if m['id'] == 'mistral-7b-v03')
    inputs = ['monolith.go', 'go.mod', 'go.sum', 'prompt_prefill_perf_test.go']
    hashes = {name: sha(root / name) for name in inputs}
    protocol = dict(format='monolith-postload-prefill-v1', model=source['repo'], revision=source['revision'],
                    files=source['files'], source_sha256=hashes, modes=['cpu', 'trace', 'plain'],
                    tokens=275, threads=4, context=1024, weight_format='q4-mixed', output_tokens=1,
                    scope='One fresh process and one Generate per mode. Only plain is uninstrumented. '
                          'All exclude loading. The trace has aggregate wait, not subtractable wall time.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    def verify_model():
        for name, digest in source['files'].items():
            if sha(model / name) != digest:
                raise ValueError('model file differs: ' + name)

    verify_model()
    goroot = subprocess.check_output([root / 'scripts/go.sh', 'env', 'GOROOT'], cwd=root, env=env, text=True)
    go = str(Path(goroot.strip()) / 'bin/go')
    build = out / 'build'
    build.mkdir()
    for name in inputs:
        (build / name).write_bytes((root / name).read_bytes())
    binary = build / 'probe.test'
    subprocess.run([go, 'test', '-c', '-trimpath', '-o', binary, '.'], cwd=build, env=env, check=True)
    binary_hash = sha(binary)
    completion = None
    reports = []
    artifacts = {}

    def render(name, command):
        path = out / name
        with path.open('xb') as f:
            subprocess.run(command, cwd=root, env=env, stdout=f, check=True)
        artifacts[name] = sha(path)

    for mode in protocol['modes']:
        report = out / (mode + '.json')
        capture = out / (mode + '.raw')
        run_env = dict(env, MONOLITH_STACK_MODEL=str(model), MONOLITH_STACK_ORDER='long',
                       MONOLITH_STACK_CAPTURE=mode, MONOLITH_STACK_CAPTURE_PATH=str(capture),
                       MONOLITH_STACK_OUTPUT=str(report))
        print('starting', mode, flush=True)
        with (out / (mode + '.log')).open('x') as log:
            resources = run_measured([binary, '-test.run=^TestHFPromptStackProbe$', '-test.v',
                                      '-test.timeout=13m'], timeout=810, cwd=root, env=run_env,
                                     stdout=log, stderr=subprocess.STDOUT)
        data = json.loads(report.read_text())
        cases = data['cases']
        if (data['threads'] != 4 or data['plan']['weight_format'] != 'q4-mixed' or
                len(cases) != 1 or cases[0]['name'] != 'long' or len(cases[0]['trials']) != 1):
            raise ValueError('unexpected probe shape')
        result = cases[0]['trials'][0]['completion']
        if result['prompt_tokens'] != 275 or result['completion_tokens'] != 1:
            raise ValueError('unexpected token counts')
        if completion is not None and completion != result:
            raise ValueError('diagnostic changed completion')
        completion = result
        reports.append(dict(mode=mode, report=data, process_resources=resources))
        if mode != 'plain':
            artifacts[capture.name] = sha(capture)
        if mode == 'cpu':
            render('cpu-top.txt', [go, 'tool', 'pprof', '-top', '-nodefraction=0', binary, capture])
            render('cpu-q4.txt', [go, 'tool', 'pprof', '-list=linearQ4Into.func1', binary, capture])
        if mode == 'trace':
            for kind in ('sched', 'sync'):
                render(kind + '.pprof', [go, 'tool', 'trace', '-pprof=' + kind, capture])
                render(kind + '-top.txt', [go, 'tool', 'pprof', '-top', '-nodefraction=0',
                                           out / (kind + '.pprof')])
        print('finished', mode, flush=True)
    verify_model()
    if sha(binary) != binary_hash or any(sha(root / n) != h or sha(build / n) != h for n, h in hashes.items()):
        raise ValueError('source or binary changed during measurement')
    receipt = dict(protocol=protocol, host=platform.system() + '/' + platform.machine(),
                   go_version=subprocess.check_output([go, 'version'], env=env, text=True).strip(),
                   binary_sha256=binary_hash, artifacts_sha256=artifacts, reports=reports,
                   exact_completion=True)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
