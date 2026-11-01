#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Apply the frozen four-row Q4 gates to two completed comparison receipts."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

from compare_inference import numerical_signature, process_schedule
from quant_quality import sha


def assess(plan, prompt, decode, exact):
    variants = ('before', 'after', 'after', 'before')
    pinned = plan['model_catalog_entry']
    if (exact['passed'] is not True or exact['sources'] != plan['exact_source_sha256']
            or sorted(exact['tests']) != sorted(plan['exact_tests']) or not re.fullmatch('[0-9a-f]{64}', exact['log_sha256'])
            or exact['modules'] != {n: plan['comparison_input_sha256'][n] for n in ('go.mod', 'go.sum')}):
        raise ValueError('candidate exactness tests did not pass')
    if (prompt['protocol']['baseline'] != plan['baseline'] or decode['baseline_commit'] != plan['baseline']
            or not prompt['files'] == decode['files'] == pinned['files'] or not prompt['revision'] == decode['revision'] == pinned['revision']
            or not prompt['model'] == decode['model'] == pinned['repo']
            or prompt['protocol']['weight_format'] != plan['weight_format']
            or decode['weight_format'] != plan['weight_format'] or decode['case_order_protocol'] != 'balanced'
            or tuple(r['variant'] for r in prompt['reports']) != variants
            or tuple(r['variant'] for r in decode['reports']) != variants):
        raise ValueError('measurement protocols differ')
    for variant in ('before', 'after'):
        expected = {n: plan['comparison_input_sha256'][n] for n in ('go.mod','go.sum')}
        expected['monolith.go'] = plan['runtime_sha256'][variant]
        if prompt['builds'][variant]['inputs_sha256'] != {**expected, 'prompt_real_test.go': plan['comparison_input_sha256']['prompt_prefill_perf_test.go']}:
            raise ValueError('wrong prompt build inputs')
        if decode['builds'][variant]['inputs_sha256'] != {**expected, 'inference_perf_test.go': plan['comparison_input_sha256']['inference_perf_test.go']}:
            raise ValueError('wrong decode build inputs')
    if decode['source_inputs_sha256']['validation/quality/corpus.json'] != plan['comparison_input_sha256']['validation/quality/corpus.json']:
        raise ValueError('wrong decode corpus')
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
            if (len(case['prompt_ids']) != case['sessions'] or any(len(ids) != 8 for ids in case['prompt_ids'])):
                raise ValueError('wrong decode input lengths')
            for t in case['trials']:
                if (any(not re.fullmatch('[0-9a-f]{64}', t[k]) for k in ('logits_sha256','kv_sha256'))
                        or len(t['token_ids']) != case['sessions'] or any(len(ids) != 8 for ids in t['token_ids'])
                        or any(not math.isfinite(v) or v <= 0 for v in t['decode_step_seconds'])
                        or not math.isclose(t['decode_seconds'], sum(t['decode_step_seconds']), rel_tol=1e-12)):
                    raise ValueError('invalid decode durations, IDs or bit digests')
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
    for name in ('protocol', 'prompt', 'decode', 'exact', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    inputs = dict(protocol=args.protocol, prompt=args.prompt, decode=args.decode, exact=args.exact)
    hashes = {n: sha(p) for n, p in inputs.items() if p.is_file()}
    try:
        exact = json.loads(args.exact.read_text())
        if exact['protocol_sha256'] != hashes['protocol'] or sha(args.exact.with_name('go-test.jsonl')) != exact['log_sha256']:
            raise ValueError('wrong exactness protocol or Go test output')
        result = assess(*(json.loads(p.read_text()) for p in inputs.values()))
        result['status'] = 'adopt' if result['accepted'] else 'reject'
    except (KeyError, TypeError, OSError, ValueError) as err:
        result = dict(status='incomplete', accepted=False, error=str(err))
    result['input_sha256'] = hashes
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    if not result['accepted']:
        raise SystemExit(2 if result['status'] == 'reject' else 1)


if __name__ == '__main__':
    main()
