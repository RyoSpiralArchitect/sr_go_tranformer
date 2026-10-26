#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Compare two monolith runtime sources with an identical bounded offline probe.

Builds the selected Git baseline and working source into separate Go test
executables. Runs only one resident model at a time in AB/BA order. Source
files must remain immutable until the receipt is complete. No downloads.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

from quant_quality import run_measured, sha


def numerical_signature(report):
    """Keep the input contract, all logit/KV bits and generated IDs, not timing."""
    return dict(plan=report['plan'], threads=report['threads'],
                kernel_workers=report['kernel_workers'], context=report['context'],
                prefill_tokens_per_session=report['prefill_tokens_per_session'],
                decode_steps=report['decode_steps'], generated_ids_per_session=report['generated_ids_per_session'],
                cases=[dict(sessions=c['sessions'], prompt_ids=c['prompt_ids'],
                            trials=[{k: t[k] for k in ('logits_sha256', 'kv_sha256', 'token_ids', 'workspace_bytes')}
                                    for t in c['trials']]) for c in report['cases']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--id', required=True, help='Pinned model catalog ID')
    parser.add_argument('--baseline-ref', required=True, help='Local Git commit to compare')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model, output = args.model.resolve(), args.output_dir.resolve()
    baseline = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options',
                                        args.baseline_ref+'^{commit}'], cwd=root, text=True).strip()
    catalog_path = root/'validation/hf/model-profiles.json'
    source = next(x for x in json.loads(catalog_path.read_text())['models'] if x['id'] == args.id)
    for name, expected in source['files'].items():
        if sha(model/name) != expected:
            raise RuntimeError('Pinned source mismatch: '+name)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [root/'monolith.go', root/'go.mod', root/'go.sum', root/'inference_perf_test.go',
              root/'validation/quality/corpus.json', catalog_path,
              Path(__file__), root/'scripts/quant_quality.py', root/'scripts/go.sh']
    source_hashes = {str(p.relative_to(root)): sha(p) for p in inputs}
    env = dict(os.environ, HF_HUB_OFFLINE='1', GOMAXPROCS='4', CGO_ENABLED='0',
               MONOLITH_PERF_MODEL=str(model), MONOLITH_PERF_CORPUS=str(root/'validation/quality/corpus.json'))
    env.pop('HF_TOKEN', None)
    # Avoid ambient flags/workspaces changing either build, or a hidden runtime
    # tuning setting making the checked-in protocol hard to reproduce.
    env.update(GOWORK='off', GOFLAGS='')
    for key in ('GOGC', 'GOMEMLIMIT', 'GODEBUG'):
        env.pop(key, None)
    # scripts/go.sh intentionally changes to the repository root. Resolve its
    # toolchain once, then invoke Go directly in each isolated build directory.
    goroot = subprocess.check_output([str(root/'scripts/go.sh'), 'env', 'GOROOT'],
                                     cwd=root, env=env, text=True).strip()
    go = str(Path(goroot)/'bin/go')
    builds = {}
    for variant in ('before', 'after'):
        build = output/variant
        build.mkdir()
        for name in ('monolith.go', 'go.mod', 'go.sum'):
            contents = (subprocess.check_output(['git', 'show', baseline+':'+name], cwd=root)
                        if variant == 'before' else (root/name).read_bytes())
            (build/name).write_bytes(contents)
        # The harness is identical for both runtimes and compiles by itself.
        (build/'inference_perf_test.go').write_bytes((root/'inference_perf_test.go').read_bytes())
        binary = build/'probe.test'
        with (build/'build.log').open('w') as log:
            subprocess.run([go, 'test', '-c', '-trimpath', '-o', str(binary), '.'],
                           cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        builds[variant] = dict(binary_sha256=sha(binary),
                               inputs_sha256={n: sha(build/n) for n in ('monolith.go', 'go.mod', 'go.sum', 'inference_perf_test.go')})
    # Two process pairs reverse run order. Each process warms both cases, then
    # measures three trials per case. Loading is outside the forward intervals.
    signature, reports = None, []
    for index, variant in enumerate(('before', 'after', 'after', 'before')):
        build = output/variant
        result = output/(str(index)+'-'+variant+'.json')
        env['MONOLITH_PERF_OUTPUT'] = str(result)
        with (output/(str(index)+'-'+variant+'.log')).open('w') as log:
            resources = run_measured([str(build/'probe.test'), '-test.run=^TestHFInferencePerformance$',
                                      '-test.v', '-test.timeout=11m'], cwd=root, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=700)
        report = json.loads(result.read_text())
        current = numerical_signature(report)
        if signature is None:
            signature = current
        elif signature != current:
            raise RuntimeError('Runtime variants changed input, logits, KV, IDs or workspace bytes')
        reports.append(dict(variant=variant, process_index=index, report=report, process_resources=resources))
        print(index, variant, 'complete; identical numerical signature', flush=True)
    if source_hashes != {str(p.relative_to(root)): sha(p) for p in inputs}:
        raise RuntimeError('Source or protocol changed during comparison')
    for name, expected in source['files'].items():
        if sha(model/name) != expected:
            raise RuntimeError('Pinned model changed during comparison: '+name)
    for variant, manifest in builds.items():
        if sha(output/variant/'probe.test') != manifest['binary_sha256']:
            raise RuntimeError('Probe executable changed')
        if {n: sha(output/variant/n) for n in manifest['inputs_sha256']} != manifest['inputs_sha256']:
            raise RuntimeError('Build inputs changed')
    receipt = dict(format='monolith-runtime-comparison-v1', baseline_commit=baseline,
                   model=source['repo'], revision=source['revision'], files=source['files'],
                   host=platform.system()+'/'+platform.machine(),
                   go_version=subprocess.check_output([go, 'version'], cwd=root, env=env, text=True).strip(),
                   hf_hub_offline=True, builds=builds, source_inputs_sha256=source_hashes,
                   numerical_identity=True, reports=reports,
                   environment=dict(GOMAXPROCS='4', CGO_ENABLED='0', GOWORK='off', GOFLAGS='',
                                    GOGC=None, GOMEMLIMIT=None, GODEBUG=None),
                   note='One resident mixed model; fixed 8-token prefixes, 7 decode calls, 1/2 sessions. Core forward timing only; peak RSS spans each full process. Not HTTP throughput or a language-quality benchmark.')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
