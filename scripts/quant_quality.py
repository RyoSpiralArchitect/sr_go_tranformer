#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Run the bounded Go quality probe offline, with one resident format at a time.

Builds a fresh Go probe in the output directory. Source files must remain
immutable throughout. The small authored corpus measures drift, not capability.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import platform


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--id', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--reference', choices=('f32', 'q8'), default='f32')
    p.add_argument('--formats', nargs='+', choices=('f32', 'q8', 'q4', 'q4-mixed'), default=['f32', 'q8', 'q4', 'q4-mixed'])
    p.add_argument('--token-inputs', type=Path, help='pinned external token IDs for profiles without a Go text tokenizer')
    p.add_argument('--weight-mib', type=int, default=8192)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--documents', type=int, default=6)
    a = p.parse_args()
    if not 1 <= a.weight_mib <= 16384 or not 1 <= a.steps <= 32 or not 1 <= a.documents <= 6 or len(set(a.formats)) != len(a.formats) or a.formats[0] != a.reference:
        p.error('bounded steps/documents required; unique formats must start with the reference')
    root = Path(__file__).resolve().parents[1]
    model, output = a.model.resolve(), a.output_dir.resolve()
    catalog = json.loads((root / 'validation/hf/model-profiles.json').read_text())
    source = next(item for item in catalog['models'] if item['id'] == a.id)
    for name, expected in source['files'].items():
        if sha(model / name) != expected:
            raise RuntimeError('Pinned source mismatch: ' + name)
    token_input_sha = None
    if a.token_inputs:
        if a.token_inputs.stat().st_size > 1 << 20:
            raise RuntimeError('External token input file exceeds 1 MiB')
        token_input_sha = sha(a.token_inputs)
        token_input = json.loads(a.token_inputs.read_text())
        if token_input['tokenizer_sha256'] != source['files'].get('tokenizer.json'):
            raise RuntimeError('External token input tokenizer identity mismatch')
    output.mkdir(parents=True, exist_ok=True)
    reference = output / 'reference.f32'
    if reference.exists():
        raise RuntimeError('Use a fresh output directory; reference already exists')
    inputs = sorted(root.glob('*.go')) + [root/'go.mod', root/'go.sum']
    build_inputs = {path.name: sha(path) for path in inputs}
    probe = output / 'probe.test'
    build_env = dict(os.environ, CGO_ENABLED='0')
    go = str(root / 'scripts/go.sh')
    with (output / 'build.log').open('w') as log:
        subprocess.run([go, 'test', '-c', '-trimpath', '-o', str(probe), '.'],
                       cwd=root, env=build_env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    if build_inputs != {path.name: sha(path) for path in inputs}:
        raise RuntimeError('Build inputs changed during compilation')
    go_version = subprocess.check_output([go, 'version'], cwd=root, text=True).strip()
    script_sha = sha(Path(__file__))
    corpus_sha = sha(root / 'validation/quality/corpus.json')
    source_sha = sha(root / 'monolith.go')
    probe_sha = sha(probe)
    env = dict(os.environ, HF_HUB_OFFLINE='1', GOMAXPROCS='4',
               MONOLITH_QUALITY_MODEL=str(model), MONOLITH_QUALITY_SOURCE=source['repo']+'@'+source['revision'],
               MONOLITH_QUALITY_REFERENCE_FORMAT=a.reference, MONOLITH_QUALITY_REFERENCE=str(reference),
               MONOLITH_QUALITY_STEPS=str(a.steps), MONOLITH_QUALITY_DOCUMENTS=str(a.documents))
    env['MONOLITH_QUALITY_WEIGHT_MIB'] = str(a.weight_mib)
    env.pop('MONOLITH_QUALITY_TOKEN_INPUTS', None)
    if a.token_inputs:
        env['MONOLITH_QUALITY_TOKEN_INPUTS'] = str(a.token_inputs.resolve())
    env.pop('HF_TOKEN', None)
    reports = {}
    for index, fmt in enumerate(a.formats):
        result = output / (fmt + '.json')
        env.update(MONOLITH_QUALITY_FORMAT=fmt, MONOLITH_QUALITY_OUTPUT=str(result),
                   MONOLITH_QUALITY_WRITE_REFERENCE='1' if index == 0 else '0')
        with (output / (fmt + '.log')).open('w') as log:
            subprocess.run([str(probe), '-test.run=^TestHFQualityProbe$', '-test.v', '-test.timeout=22m'],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1350, check=True)
        r = json.loads(result.read_text())
        if r['source_sha256'] != source_sha or r['reference']['corpus_sha256'] != corpus_sha:
            raise RuntimeError('Source or corpus changed during probe')
        reports[fmt] = r
        print(a.id, fmt, 'NLL', r['mean_nll'], 'KL', r['mean_kl'], 'top1', r['top1_matches'], '/', r['scored_tokens'], flush=True)
    if a.token_inputs and sha(a.token_inputs) != token_input_sha:
        raise RuntimeError('External token input file changed')
    if len({r['reference_vectors_sha256'] for r in reports.values()}) != 1:
        raise RuntimeError('Compared different reference vectors')
    if (build_inputs != {path.name: sha(path) for path in inputs} or sha(probe) != probe_sha
            or sha(Path(__file__)) != script_sha):
        raise RuntimeError('Probe or source changed during evaluation')
    receipt = dict(format='monolith-quant-quality-suite-v1', model=source['repo'], revision=source['revision'],
                   files=source['files'], source_sha256=source_sha, corpus_sha256=corpus_sha,
                   probe_sha256=probe_sha, script_sha256=script_sha, build_inputs_sha256=build_inputs,
                   go_version=go_version, host=platform.system()+"/"+platform.machine(), reference_sha256=sha(reference),
                   hf_hub_offline=True, reference_format=a.reference, steps=a.steps, documents=a.documents,
                   weight_mib=a.weight_mib, external_token_inputs_sha256=token_input_sha,
                   note='Original small corpus, held out from policy selection; no general quality or cross-model perplexity claim.', reports=reports)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
