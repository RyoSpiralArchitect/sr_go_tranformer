#!/usr/bin/env python3
"""Bounded CLI smoke against an existing hash-bound real-model receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import selectors
import signal
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ref = json.loads(a.reference.read_text())
    for name, sha in ref['files'].items():
        if digest(a.model / name) != sha:
            raise RuntimeError('Model identity mismatch: ' + name)
    binary = str(a.binary.resolve())
    base = ['-hf-model', str(a.model)]
    checked = json.loads(subprocess.check_output([binary, 'check-hf'] + base + ['-chat', '-json'], text=True, timeout=30))
    if checked['weights']['config']['context'] != 2048 or checked['weight_values_checked']:
        raise RuntimeError('Incorrect default context or payload assurance')
    case = next(c for c in ref['cases'] if 'messages' in c['request'])
    prompt = case['request']['messages'][0]['content']
    flags = base + ['-threads','4','-max-tokens','8','-temperature','0','-repeat-penalty','1']
    for source, data in [(['-chat','-prompt-file','-'], prompt),
                         (['-messages','-'], json.dumps(case['request']['messages']))]:
        actual = json.loads(subprocess.check_output([binary,'generate']+flags+source+['-json'], input=data, text=True, timeout=240))
        if actual != case['completion']:
            raise RuntimeError('stdin generation mismatch')
    actual = subprocess.check_output([binary,'chat']+flags+['-kernel-workers','2'],
              input=prompt+'\n/reset\n'+prompt+'\n/exit\n', text=True, stderr=subprocess.PIPE, timeout=240)
    if actual != (case['completion']['text']+'\n')*2:
        raise RuntimeError('Interactive chat/reset mismatch')
    # Leave stdin open and empty, then interrupt a model awaiting input.
    proc = subprocess.Popen([binary,'chat']+flags+['-kernel-workers','2'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic()+45
        ready = b''
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stderr,selectors.EVENT_READ)
            while b'Chat ready.' not in ready:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Interactive startup failed')
                if selector.select(timeout=0.5):
                    ready += proc.stderr.read1(4096)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=8)
        stdout, stderr = proc.communicate(timeout=1)
        if proc.returncode != 1 or stdout or b'context canceled' not in stderr:
            raise RuntimeError('stdin cancellation did not terminate cleanly')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    a.output.write_text(json.dumps(dict(format='monolith-hf-cli-smoke-v1', model=ref['model'],
        revision=ref['revision'], files=ref['files'], binary_sha256=digest(a.binary),
        reference_sha256=digest(a.reference), preflight=checked, prompt_stdin_match=True,
        messages_stdin_match=True, interactive_reset_match=True, idle_stdin_sigint_exit=True), indent=2)+'\n')
    print('PASS: preflight, default budgets, stdin text/JSON, interactive reset, idle SIGINT')


if __name__ == '__main__':
    main()
