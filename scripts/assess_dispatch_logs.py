import argparse
import json
import re
from pathlib import Path

from compare_fp32_dispatch import summarize
from quant_quality import sha


EXPRESSION = re.compile(r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+([\d.]+) B/op\s+([\d.]+) allocs/op$', re.MULTILINE)


def read_samples(path):
    text = path.read_text()
    if not re.search(r'^PASS$', text, re.MULTILINE) or re.search(r'^(FAIL|panic:)', text, re.MULTILINE):
        raise ValueError('benchmark process did not pass: ' + str(path))
    samples = [dict(name=m[0], ns_per_op=float(m[1]), bytes_per_op=float(m[2]),
                    allocs_per_op=float(m[3])) for m in EXPRESSION.findall(text)]
    if not samples or any(s['ns_per_op'] <= 0 for s in samples):
        raise ValueError('missing or invalid benchmark samples: ' + str(path))
    return samples


def log_header(path):
    text = path.read_text()
    header = {}
    for field in ('goos', 'goarch', 'cpu'):
        values = re.findall(r'^' + field + r': (.+)$', text, re.MULTILINE)
        if len(values) != 1:
            raise ValueError('missing or repeated benchmark header: ' + str(path))
        header[field] = values[0]
    return header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline-label', default='before')
    parser.add_argument('--before-build', type=Path, required=True)
    parser.add_argument('--after-build', type=Path, required=True)
    parser.add_argument('--before-binary', type=Path, required=True)
    parser.add_argument('--after-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    builds = {}
    for variant, directory, binary in (
        ('before', args.before_build, args.before_binary),
        ('after', args.after_build, args.after_binary),
    ):
        files = [*directory.glob('*.go'), directory / 'go.mod', directory / 'go.sum']
        builds[variant] = dict(binary_sha256=sha(binary), inputs_sha256={p.name: sha(p) for p in files})
    before = {n: h for n, h in builds['before']['inputs_sha256'].items() if n != 'monolith.go'}
    after = {n: h for n, h in builds['after']['inputs_sha256'].items() if n != 'monolith.go'}
    if before != after or not before:
        raise ValueError('comparison did not use identical shared sources')
    reports = []
    expected_cases = None
    expected_header = None
    for index, label in enumerate((args.baseline_label, args.candidate, args.candidate, args.baseline_label)):
        current_cases = {}
        for cohort in ('linear', 'generate', 'quant'):
            path = args.logs / f'{args.prefix}-{index}-{label}-{cohort}.txt'
            samples = read_samples(path)
            header = log_header(path)
            if expected_header is None:
                expected_header = header
            if header != expected_header:
                raise ValueError('mixed benchmark host or target headers')
            reports.append(dict(variant='before' if label == args.baseline_label else 'after',
                                process_index=index, cohort=cohort, samples=samples,
                                log_sha256=sha(path), log_name=path.name))
            for sample in samples:
                current_cases[sample['name']] = current_cases.get(sample['name'], 0) + 1
        if expected_cases is None:
            expected_cases = current_cases
        if current_cases != expected_cases or any(count != 2 for count in current_cases.values()):
            raise ValueError('missing, duplicate or unbalanced benchmark cases')
    receipt = dict(format='monolith-imported-dispatch-comparison-v1', builds=builds,
                   host=expected_header, reports=reports, summary=summarize(reports),
                   scope='ABBA with two samples per case per process. Read toolchain, affinity and timing settings from the associated build/experiment index; this importer does not infer them from timing text. Synthetic fixtures only, not a real-model speed claim.')
    with args.output.open('x') as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write('\n')
    for name, row in receipt['summary'].items():
        print(f"{name}: time ratio {row['time_ratio']:.3f}, allocations {row['before']['allocs_per_op']:g} -> {row['after']['allocs_per_op']:g}")


if __name__ == '__main__':
    main()
