#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Apply the frozen four-row Q4 gates to two completed comparison receipts."""
import argparse
import json
import math
import statistics
from pathlib import Path

from compare_inference import numerical_signature, process_schedule
from quant_quality import sha


def assess(plan, prompt, decode):
    variants = ('before', 'after', 'after', 'before')
    if (prompt['protocol']['baseline'] != plan['baseline'] or decode['baseline_commit'] != plan['baseline']
            or prompt['files'] != decode['files'] or prompt['revision'] != decode['revision']
            or prompt['protocol']['weight_format'] != plan['weight_format']
            or decode['weight_format'] != plan['weight_format'] or decode['case_order_protocol'] != 'balanced'
            or tuple(r['variant'] for r in prompt['reports']) != variants
            or tuple(r['variant'] for r in decode['reports']) != variants):
        raise ValueError('measurement protocols differ')
    for variant in ('before', 'after'):
        if prompt['builds'][variant]['inputs_sha256']['monolith.go'] != decode['builds'][variant]['inputs_sha256']['monolith.go']:
            raise ValueError('different prompt and decode runtimes')
    sig = numerical_signature(decode['reports'][0]['report'])
    if not all(numerical_signature(r['report']) == sig for r in decode['reports']):
        raise ValueError('logits, used KV, IDs, plan or workspace differ')
    samples = {case: {'before': [], 'after': []} for case in ('short', 'long', 'decode_1', 'decode_2')}
    completions = {}
    for proc, order in zip(prompt['reports'], ['short,long','short,long','long,short','long,short']):
        report = proc['report']
        if report['threads'] != plan['threads'] or report['plan']['weight_format'] != plan['weight_format'] or report['plan']['config']['context'] != plan['context'] or [c['name'] for c in report['cases']] != order.split(','):
            raise ValueError('prompt shape or order differs')
        for case in report['cases']:
            name = case['name']
            count, tokens = (3, 4) if name == 'short' else (1, 275)
            if len(case['trials']) != count:
                raise ValueError('missing prompt samples')
            for trial in case['trials']:
                result = trial['completion']
                if result['prompt_tokens'] != tokens or result['completion_tokens'] != 1 or completions.setdefault(name, result) != result:
                    raise ValueError('completion mismatch')
                samples[name][proc['variant']].append(trial['seconds'])
    for proc, (_, order) in zip(decode['reports'], process_schedule('balanced')):
        report = proc['report']
        if (report['case_order'] != order or [c['sessions'] for c in report['cases']] != order
                or report['threads'] != plan['threads'] or report['kernel_workers'] != plan['threads']
                or report['warmup_runs_per_case'] != 1 or report['decode_steps'] != 7):
            raise ValueError('decode shape or order differs')
        for case in report['cases']:
            if len(case['trials']) != 3 or any(len(t['decode_step_seconds']) != 7 for t in case['trials']):
                raise ValueError('missing decode samples')
            samples['decode_' + str(case['sessions'])][proc['variant']].extend(t['decode_seconds'] for t in case['trials'])
    results = {}
    for case, values in samples.items():
        if any(not math.isfinite(t) or t <= 0 for trials in values.values() for t in trials):
            raise ValueError('invalid elapsed time')
        before, after = (statistics.median(values[v]) for v in ('before', 'after'))
        limit = plan['timing'].get(case + '_max_ratio', plan['decode_controls']['max_median_ratio'])
        results[case] = dict(before=before, after=after, ratio=after/before, limit=limit, passed=after/before <= limit)
    return dict(accepted=all(r['passed'] for r in results.values()), cases=results, numerical_identity=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'prompt', 'decode', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.protocol, args.prompt, args.decode]
    result = assess(*(json.loads(p.read_text()) for p in inputs))
    result['input_sha256'] = dict(zip(('protocol', 'prompt', 'decode'), (sha(p) for p in inputs)))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
