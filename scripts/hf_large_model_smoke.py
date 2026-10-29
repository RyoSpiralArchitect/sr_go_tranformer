#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Bounded offline CLI/HTTP/concurrent-SSE smoke for one resident HF format.

Verifies pinned source files and builds the Go server/load client. This is a
functional/resource test, not a throughput comparison or quality benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import time
from urllib.error import URLError

from quant_quality import sha
from q8_reference import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--format', choices=('q8', 'q4', 'q4-mixed'), default='q4-mixed')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model, output = args.model.resolve(), args.output_dir.resolve()
    source = next(x for x in json.loads((root/'validation/hf/model-profiles.json').read_text())['models'] if x['id'] == args.id)
    for name, expected in source['files'].items():
        if sha(model/name) != expected:
            raise RuntimeError('Pinned source mismatch: '+name)
    output.mkdir(parents=True, exist_ok=False)
    inputs = sorted(root.glob('*.go')) + sorted((root/'tools/loadtest').glob('*.go')) + [root/'go.mod', root/'go.sum']
    build_hashes = {str(p.relative_to(root)): sha(p) for p in inputs}
    script_inputs = {p.name: sha(p) for p in (Path(__file__), root/'scripts/quant_quality.py', root/'scripts/q8_reference.py')}
    env = dict(os.environ, HF_HUB_OFFLINE='1', GOMAXPROCS='4', CGO_ENABLED='0')
    env.pop('HF_TOKEN', None)
    binary, client = output/'monolith', output/'loadtest'
    with (output/'build.log').open('w') as log:
        for target, package in ((binary, '.'), (client, './tools/loadtest')):
            subprocess.run([str(root/'scripts/go.sh'), 'build', '-trimpath', '-o', str(target), package],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    flags = ['infer', 'local', '-path', str(model), '-context', '128', '-weight-format', args.format, '-weight-mib', '8192']
    def run(action, extra):
        return json.loads(subprocess.check_output([str(binary)]+flags+[action]+extra, env=env, timeout=600))
    plan = run('inspect', [])
    budget = math.ceil(plan['weight_bytes']/(1 << 20))
    flags[-1] = str(budget)
    preflight = run('check', ['-chat', '-json'])
    requests = [dict(prompt='The capital of France is'),
                dict(messages=[dict(role='user', content='Write one short sentence about cats.')])]
    cases = []
    for body in requests:
        prompt_flags = ['-prompt', body['prompt']] if 'prompt' in body else ['-chat', '-prompt', body['messages'][0]['content']]
        got = run('generate', prompt_flags+['-threads', '4', '-max-tokens', '8', '-temperature', '0', '-repeat-penalty', '1', '-json'])
        cases.append(dict(request=body, completion=got))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = 'http://127.0.0.1:'+str(port)
    with (output/'server.log').open('w') as log:
        started = time.monotonic()
        proc = subprocess.Popen([str(binary)]+flags+['serve',
            '-addr', '127.0.0.1:'+str(port), '-threads', '4', '-kernel-workers', '4',
            '-max-batch', '2', '-queue', '4', '-kv-mib', '32', '-prefill-chunk', '8',
            '-mixed-prefill-chunk', '4', '-token-budget', '16'], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Server startup failed; inspect server.log')
                try:
                    request(base+'/healthz')
                    break
                except URLError:
                    if time.monotonic()-started > 600:
                        raise RuntimeError('Server startup timeout') from None
                    time.sleep(.1)
            startup = time.monotonic()-started
            def complete(case, streaming):
                return request(base+'/v1/completions', dict(case['request'], max_tokens=8,
                    temperature=0, repeat_penalty=1, stream=streaming))
            for case in cases:
                if complete(case, False) != case['completion']:
                    raise RuntimeError('CLI/HTTP mismatch')
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda c: complete(c, True), cases))
            if results != [case['completion'] for case in cases]:
                raise RuntimeError('CLI/concurrent SSE mismatch')
            command = [str(client), '-url', base, '-requests', '2', '-concurrency', '2',
                '-max-tokens', '8', '-short-prompt', 'The capital of France is',
                '-long-prompt', 'Once upon a time, a small cat', '-cancel-every', '2',
                '-cancel-after', '1', '-timeout', '120s', '-sample-interval', '100ms', '-drain-timeout', '20s']
            report = json.loads(subprocess.check_output(command, env=env, timeout=300))
            if report['outcomes'] != {'completed': 1, 'canceled': 1}:
                raise RuntimeError('Unexpected completion/cancellation outcome: '+str(report['outcomes']))
            if report['server_metrics']['counter_deltas'].get('monolith_canceled_total') != 1:
                raise RuntimeError('Client cancellation was not observed by the server')
            if report['server_metrics']['drain_status'] != 'drained':
                raise RuntimeError('Engine did not drain')
            stats = request(base+'/healthz')['stats']
            if any(stats[k] != 0 for k in ('active', 'queue', 'computing', 'admission_used', 'kv_reserved_bytes')) or stats['submitted'] != stats['released']:
                raise RuntimeError('Resources not released')
            rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(proc.pid)], text=True))*1024
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise RuntimeError('Shutdown timeout')
        if proc.returncode != 0:
            raise RuntimeError('Unclean server shutdown')
    if script_inputs != {p.name: sha(p) for p in (Path(__file__), root/'scripts/quant_quality.py', root/'scripts/q8_reference.py')}:
        raise RuntimeError('Scripts changed during smoke')
    if build_hashes != {str(p.relative_to(root)): sha(p) for p in inputs}:
        raise RuntimeError('Source changed during smoke')
    receipt = dict(format='monolith-large-model-smoke-v1', model=source['repo'], revision=source['revision'],
        files=source['files'], weight_format=args.format, weight_mib=budget, plan=plan, preflight=preflight,
        hf_hub_offline=True, entry_point='infer local -path', cases=cases, cli_http_concurrent_sse_match=True,
        cancellation_load=report, final_engine_stats=stats, clean_shutdown=True, startup_seconds=startup,
        post_smoke_rss_bytes=rss, rss_method='one post-smoke ps observation; not peak',
        binary_sha256=sha(binary), loadtest_sha256=sha(client), build_inputs_sha256=build_hashes,
        script_inputs_sha256=script_inputs, host=platform.system()+'/'+platform.machine(),
        go_version=subprocess.check_output([str(root/'scripts/go.sh'), 'version'], cwd=root, text=True).strip(),
        server=dict(context=128, threads=4, kernel_workers=4, max_batch=2, queue=4, kv_mib=32,
                    prefill_chunk=8, mixed_prefill_chunk=4, token_budget=16),
        note='Bounded functional/lifecycle trial; timings are not a throughput comparison.')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(args.id, args.format, 'CLI/HTTP/SSE matched; cancellation workload drained; clean shutdown', flush=True)


if __name__ == '__main__':
    main()
