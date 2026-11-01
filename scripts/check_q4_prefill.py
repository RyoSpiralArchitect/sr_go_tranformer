#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Run the candidate-bound exact Q4 tests and retain their Go JSON event log."""
import argparse
import json
import re
import subprocess
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    protocol = json.loads(args.protocol.read_text())

    def hashes():
        return {p.name: sha(p) for p in sorted(root.glob('*.go'))}

    sources = hashes()
    inputs = {n: sha(root / n) for n in ('go.mod', 'go.sum')}
    result = dict(passed=False, sources=sources, modules=inputs, tests=[], protocol_sha256=sha(args.protocol))
    log = out / 'go-test.jsonl'
    try:
        if sources != protocol['exact_source_sha256'] or any(protocol['comparison_input_sha256'][n] != h for n, h in inputs.items()):
            raise ValueError('not the frozen candidate source')
        names = protocol['exact_tests']
        env = dict(isolated_go_environment(), CGO_ENABLED='1')
        pattern = '^(' + '|'.join(re.escape(n) for n in names) + ')$'
        with log.open('x') as dst:
            run = subprocess.run([root / 'scripts/go.sh', 'test', '-json', '-race', '-count=1',
                                  '-run', pattern, '.'], cwd=root, env=env, stdout=dst, timeout=300)
        events = [json.loads(line) for line in log.read_text().splitlines()]
        passed = sorted(e['Test'] for e in events if e.get('Action') == 'pass' and e.get('Test') in names)
        result['tests'] = passed
        if sources != hashes():
            raise ValueError('candidate changed during exactness check')
        result['failed_tests'] = sorted(e['Test'] for e in events if e.get('Action') == 'fail' and e.get('Test') in names)
        if run.returncode or passed != sorted(names):
            raise ValueError('one or more exactness tests did not pass on the frozen sources')
        result['passed'] = True
    except (ValueError, OSError, subprocess.SubprocessError) as err:
        result['error'] = str(err)
    if log.is_file():
        result['log_sha256'] = sha(log)
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise SystemExit(result['error'])


if __name__ == '__main__':
    main()
