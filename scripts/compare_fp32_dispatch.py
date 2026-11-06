import argparse
import json
import platform
import re
import statistics
import subprocess
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import git_source, run_measured, sha


def summarize(reports):
    cases = {}
    for report in reports:
        for sample in report['samples']:
            cases.setdefault(sample['name'], {}).setdefault(report['variant'], []).append(sample)
    summary = {}
    for name, variants in cases.items():
        row = {}
        for variant in ('before', 'after'):
            samples = variants[variant]
            row[variant] = {
                field: statistics.median(sample[field] for sample in samples)
                for field in ('ns_per_op', 'bytes_per_op', 'allocs_per_op')
            }
            row[variant]['time_range'] = [min(s['ns_per_op'] for s in samples),
                                          max(s['ns_per_op'] for s in samples)]
        row['time_ratio'] = row['after']['ns_per_op'] / row['before']['ns_per_op']
        summary[name] = row
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchtime', default='300ms')
    parser.add_argument('--count', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.count <= 10:
        parser.error('--count must be in 1..10')
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = isolated_go_environment()
    baseline = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', args.baseline_ref + '^{commit}'],
        cwd=root, text=True
    ).strip()
    goroot = subprocess.check_output([root / 'scripts/go.sh', 'env', 'GOROOT'],
                                     cwd=root, env=env, text=True).strip()
    go = str(Path(goroot) / 'bin/go')
    sources = [root / 'go.mod', root / 'go.sum', *sorted((root / 'cmd/monolith').glob('*.go'))]
    hashes = {str(path.relative_to(root)): sha(path) for path in sources}
    shared = [path for path in sources if path.name not in ('monolith.go', 'executor_experiments_test.go')]
    builds = {}
    for variant in ('before', 'after'):
        build = out / variant
        build.mkdir()
        for path in shared:
            (build / path.name).write_bytes(path.read_bytes())
        source = git_source(root, baseline, 'monolith.go') if variant == 'before' else (
            root / 'cmd/monolith/monolith.go').read_bytes()
        (build / 'monolith.go').write_bytes(source)
        binary = build / 'probe.test'
        with (build / 'build.log').open('x') as log:
            subprocess.run([go, 'test', '-c', '-trimpath', '-o', binary, '.'], cwd=build,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        builds[variant] = dict(binary_sha256=sha(binary),
                               inputs_sha256={p.name: sha(p) for p in build.glob('*.go')})
    patterns = {
        'linear': '^BenchmarkInferenceLinearWorkers$/^rows(1|4|64)$/^reuse$',
        'prefill': '^BenchmarkPromptPrefillGenerate$/^prefix(3|129|257)$/^reversefalse$/^after$',
        'generate': '^BenchmarkOwnedGenerate$/^tokens(8|128)$/^reversefalse$/^after$',
    }
    expected = {'linear': 3, 'prefill': 3, 'generate': 2}
    expression = re.compile(r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+([\d.]+) B/op\s+([\d.]+) allocs/op$', re.MULTILINE)
    reports = []
    for index, variant in enumerate(('before', 'after', 'after', 'before')):
        for cohort, pattern in patterns.items():
            log_path = out / f'{index}-{variant}-{cohort}.txt'
            with log_path.open('x') as log:
                resources = run_measured(
                    [out / variant / 'probe.test', '-test.run=^$', '-test.bench=' + pattern,
                     '-test.benchtime=' + args.benchtime, '-test.count=' + str(args.count),
                     '-test.cpu=4', '-test.benchmem'], cwd=root, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
            samples = [dict(name=m[0], ns_per_op=float(m[1]), bytes_per_op=float(m[2]),
                            allocs_per_op=float(m[3])) for m in expression.findall(log_path.read_text())]
            if len(samples) != expected[cohort] * args.count:
                raise ValueError('incomplete benchmark cohort: ' + str(log_path))
            reports.append(dict(variant=variant, process_index=index, cohort=cohort,
                                samples=samples, resources=resources, log_sha256=sha(log_path)))
            print(index, variant, cohort, 'complete', flush=True)
    if hashes != {str(path.relative_to(root)): sha(path) for path in sources}:
        raise ValueError('source changed during measurement')
    for variant, build in builds.items():
        if sha(out / variant / 'probe.test') != build['binary_sha256'] or any(
            sha(out / variant / name) != digest for name, digest in build['inputs_sha256'].items()
        ):
            raise ValueError('build changed during measurement')
    receipt = dict(format='monolith-fp32-dispatch-comparison-v1', baseline=baseline,
                   go_version=subprocess.check_output([go, 'version'], text=True).strip(),
                   host=platform.system() + '/' + platform.machine(), source_sha256=hashes,
                   driver_sha256=sha(Path(__file__)), builds=builds, reports=reports,
                   benchtime=args.benchtime, count=args.count, process_order=['before', 'after', 'after', 'before'],
                   summary=summarize(reports),
                   limitations='Synthetic FP32 fixtures; no real-model or HTTP claim. Benchmark reference checks verify completions. Timing and sampled process RSS include different scopes; source-specific allocation tests run separately.')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
